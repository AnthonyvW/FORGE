# list_frame.py
from typing import Callable, Iterable, Iterator, List, Optional, Sequence
import pygame
from UI.frame import Frame

RowBuilder = Callable[[int, Frame], None]
ElementFactory = Callable[[Frame, int], Frame]


class RowContainer(Frame):
    """A thin container for one row inside a ListFrame."""
    def __init__(self, parent: Frame, index: int, row_height: int, **kwargs):
        # Position rows at (0, index*row_height) within the ListFrame
        super().__init__(parent=parent, x=0, y=index * row_height,
                         width=1.0, height=row_height,
                         width_is_percent=True, height_is_percent=False,
                         **kwargs)
        self.index = index
        self.row_height = row_height

    def set_index_and_y(self, index: int) -> None:
        self.index = index
        self.y = index * self.row_height


class ListFrame(Frame):
    """
    A vertical list container that repeats a row blueprint N times.

    Build options:
      A) row_builder(index, row_parent): create row contents for each index
      B) element_factories: sequence of callables (parent, index) -> Frame

    Public API:
      - set_count(n), set_row_height(h), rebuild()
      - get_row(i) -> RowContainer
      - __len__, __iter__, __getitem__
      - update_row(i, rebuild: bool = False, fn: Optional[RowBuilder] = None)

    Notes:
      - All child element positions are relative to their RowContainer,
        which itself is positioned inside the ListFrame at y = index * row_height.
      - Z-ordering and input handling inherit from Frame.
    """
    def __init__(
        self,
        parent: Optional[Frame] = None,
        *,
        x: float = 0,
        y: float = 0,
        width: float = 100,
        height: float = 100,
        x_is_percent: bool = False,
        y_is_percent: bool = False,
        width_is_percent: bool = False,
        height_is_percent: bool = False,
        z_index: int = 0,
        x_align: str = "left",
        y_align: str = "top",
        background_color: Optional[pygame.Color] = None,
        row_height: int = 24,
        count: int = 0,
        row_builder: Optional[RowBuilder] = None,
        element_factories: Optional[Sequence[ElementFactory]] = None,
    ):
        super().__init__(
            parent=parent, x=x, y=y, width=width, height=height,
            x_is_percent=x_is_percent, y_is_percent=y_is_percent,
            width_is_percent=width_is_percent, height_is_percent=height_is_percent,
            z_index=z_index, x_align=x_align, y_align=y_align,
            background_color=background_color
        )
        self._row_height = int(row_height)
        self._count = int(count)
        self._row_builder: Optional[RowBuilder] = row_builder
        self._factories: Optional[List[ElementFactory]] = list(element_factories) if element_factories else None

        self._rows: List[RowContainer] = []
        if self._count > 0:
            self._materialize_rows()

    # ------------ public API ------------

    def set_count(self, n: int) -> None:
        n = max(0, int(n))
        if n == self._count:
            return
        self._count = n
        self._resize_rows()

    def set_row_height(self, h: int) -> None:
        h = max(1, int(h))
        if h == self._row_height:
            return
        self._row_height = h
        # Reposition rows and update their heights
        for i, row in enumerate(self._rows):
            row.row_height = h
            row.height = h
            row.set_index_and_y(i)

    def get_row(self, index: int) -> RowContainer:
        return self._rows[index]

    def rebuild(self) -> None:
        """Fully rebuild all rows (e.g., after changing builder/factories)."""
        self._clear_rows()
        self._materialize_rows()

    def update_row(self, index: int, rebuild: bool = False, fn: Optional[RowBuilder] = None) -> None:
        """Optionally clear and rebuild a single row with a temporary or new builder."""
        row = self._rows[index]
        if rebuild:
            # Clear children of this row
            for ch in list(row.children):
                row.children.remove(ch)
        builder = fn or self._row_builder
        if builder is not None:
            builder(index, row)
        elif self._factories:
            for f in self._factories:
                f(row, index)

    # --- Pythonic container behavior ---
    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[RowContainer]:
        return iter(self._rows)

    def __getitem__(self, i: int) -> RowContainer:
        return self.get_row(i)

    # ------------ internals ------------

    def _clear_rows(self) -> None:
        # Detach row containers from our children list
        for row in self._rows:
            if row in self.children:
                self.children.remove(row)
        self._rows.clear()

    def _materialize_rows(self) -> None:
        for i in range(self._count):
            row = RowContainer(parent=self, index=i, row_height=self._row_height)
            self._rows.append(row)
            # Build row contents
            if self._row_builder is not None:
                self._row_builder(i, row)
            elif self._factories:
                for f in self._factories:
                    f(row, i)

        # Keep our overall children z-sorted (your Frame.add_child sorts on insert,
        # but we added directly via RowContainer(parent=self), so re-sort here)
        self.children.sort(key=lambda c: c.z_index, reverse=True)

    def _resize_rows(self) -> None:
        """Grow/shrink rows to match current count; re-use existing when possible."""
        cur = len(self._rows)
        if self._count == cur:
            return

        if self._count < cur:
            # Remove extras from the end
            to_remove = self._rows[self._count:]
            for row in to_remove:
                if row in self.children:
                    self.children.remove(row)
            self._rows = self._rows[:self._count]
        else:
            # Add new rows
            for i in range(cur, self._count):
                row = RowContainer(parent=self, index=i, row_height=self._row_height)
                self._rows.append(row)
                if self._row_builder is not None:
                    self._row_builder(i, row)
                elif self._factories:
                    for f in self._factories:
                        f(row, i)

        # Reposition all rows to keep indices consistent
        for i, row in enumerate(self._rows):
            row.set_index_and_y(i)

        self.children.sort(key=lambda c: c.z_index, reverse=True)
