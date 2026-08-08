"""State tracking with checkpoint/rollback support."""

from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from binaryninja import BinaryView


@dataclass
class Checkpoint:
    """Represents a saved analysis state."""

    name: str
    timestamp: float
    # Number of wrapper-tracked mutations that had been recorded at the time
    # this checkpoint was created. Rollback replays Binary Ninja's `undo()`
    # (count_at_rollback - undo_action_count) times to revert to this point.
    # See StateTracker.create_checkpoint/rollback for why this is a count and
    # not the old `bv.undoable_actions()` stack depth.
    undo_action_count: int


class StateTracker:
    """Tracks analysis state for checkpoint/rollback support.

    Rollback works by replaying Binary Ninja's ``bv.undo()`` once per
    wrapper-tracked mutation made since the checkpoint. Each wrapper mutation
    (``rename_function``, ``set_comment``, ``define_type``, ...) performs a
    single BinaryView operation, which Binary Ninja records as exactly one
    undoable unit — so the number of ``record_change()`` calls is the number of
    ``undo()`` calls needed to revert them.

    .. note::
       Older Binary Ninja builds exposed ``bv.undoable_actions()`` so the stack
       depth could be queried directly. That API was **removed in Binary Ninja
       5.3** (replaced by the named-transaction API
       ``begin_undo_actions``/``commit_undo_actions``/``revert_undo_actions``,
       which has no depth query). Counting tracked mutations ourselves is the
       replacement, with one limitation: raw ``bv`` mutations that bypass the
       wrapper (e.g. ``bv.get_function_at(a).name = ...``) are NOT counted, so
       they won't be rolled back. Use the wrapper methods for anything you may
       want to revert.
    """

    def __init__(self, bv: "BinaryView", enabled: bool = True):
        self._bv = bv
        self._enabled = enabled
        self.checkpoints: list[Checkpoint] = []
        self.pending_changes: list[str] = []

        # Monotonic counter of wrapper-tracked mutations since the tracker was
        # created. Each wrapper method calls record_change() once after its bv
        # operation, so this equals the number of undoable units the wrapper
        # has produced.
        self._change_count = 0

        # Callback for operation logging (used by TUI)
        self.on_operation: Optional[Callable[[str], None]] = None

    def record_change(self, description: str) -> None:
        """Record a mutation for tracking.

        Called by every wrapper mutation after its BinaryView operation. Each
        call represents one undoable unit (one ``bv.undo()``), so this both
        logs the change and advances the undo-count used by rollback.
        """
        if self._enabled:
            self._change_count += 1
            self.pending_changes.append(description)
            if self.on_operation:
                self.on_operation(description)

    def create_checkpoint(self, name: str) -> bool:
        """Create named checkpoint at current state."""
        if any(cp.name == name for cp in self.checkpoints):
            return False

        # Binary Ninja 5.3 removed bv.undoable_actions(), so we can no longer
        # query the undo-stack depth. Instead we snapshot our own count of
        # wrapper-tracked mutations; rollback undoes the delta. See the class
        # docstring for the limitation (raw bv mutations aren't counted).
        self.checkpoints.append(
            Checkpoint(name=name, timestamp=time(), undo_action_count=self._change_count)
        )
        self.pending_changes.clear()
        return True

    def rollback(self, name: str) -> bool:
        """Rollback to named checkpoint by replaying ``bv.undo()``.

        Undoes ``(changes since checkpoint)`` undoable units. Returns True if
        the checkpoint was found and the undo replay was attempted, False if
        the checkpoint name is unknown. A True return does NOT guarantee every
        change reverted — only wrapper-tracked mutations are counted (see class
        docstring); raw ``bv`` mutations since the checkpoint are not undone.
        """
        checkpoint = next((cp for cp in self.checkpoints if cp.name == name), None)
        if not checkpoint:
            return False

        undo_count = self._change_count - checkpoint.undo_action_count
        for _ in range(undo_count):
            self._bv.undo()

        # Roll the counter back to the checkpoint's value: the undid changes
        # are no longer "pending since checkpoint".
        self._change_count = checkpoint.undo_action_count

        # Remove checkpoints created after this one
        self.checkpoints = [
            cp for cp in self.checkpoints if cp.timestamp <= checkpoint.timestamp
        ]
        self.pending_changes.clear()
        return True

    def get_summary(self) -> str:
        """Generate context summary for LLM."""
        if not self._enabled:
            return "# State tracking: disabled"

        lines = ["# Session state:"]

        if self.checkpoints:
            latest = self.checkpoints[-1]
            age = int(time() - latest.timestamp)
            age_str = f"{age}s ago" if age < 60 else f"{age // 60}m ago"
            lines.append(f'#   Checkpoint: "{latest.name}" ({age_str})')
        else:
            lines.append("#   Checkpoint: none")

        if self.pending_changes:
            lines.append(f"#   Pending changes: {len(self.pending_changes)}")

        lines.append(f"#   Rollback available: {'yes' if self.checkpoints else 'no'}")

        return "\n".join(lines)

    def list_checkpoints(self) -> list[dict]:
        """List all checkpoints."""
        return [{"name": cp.name, "timestamp": cp.timestamp} for cp in self.checkpoints]
