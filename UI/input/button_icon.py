import pygame
from UI.frame import Frame


def _load_surface(img_or_path: pygame.Surface | str) -> pygame.Surface:
    """Load and convert a surface from either a path or an existing Surface."""
    if isinstance(img_or_path, pygame.Surface):
        return img_or_path.convert_alpha()
    return pygame.image.load(img_or_path).convert_alpha()


def _recolor_by_alpha_mask(src: pygame.Surface,
                           fill_rgba: tuple[int, int, int, int]) -> pygame.Surface:
    """
    Recolor a monochrome/black symbol using ONLY its alpha as a mask.
    All nontransparent pixels become `fill_rgba` with their original per-pixel alpha.
    """
    out = pygame.Surface(src.get_size(), pygame.SRCALPHA)
    out.fill(fill_rgba)

    try:
        # Fast path with NumPy/surfarray
        import pygame.surfarray as sarr  # numpy-backed
        alpha_src = sarr.pixels_alpha(src).copy()  # copy to avoid locking issues
        sarr.pixels_alpha(out)[:, :] = alpha_src
    except Exception:
        # Fallback: per-pixel (a bit slower, but fine for icons)
        w, h = src.get_size()
        try:
            for y in range(h):
                for x in range(w):
                    a = src.get_at((x, y)).a
                    if a:  # only where alpha > 0
                        r, g, b, _ = fill_rgba
                        out.set_at((x, y), pygame.Color(r, g, b, a))
        finally:
            out.unlock()
            src.unlock()
    return out



class ButtonIcon(Frame):
    """
    A child widget that displays an image on top of a Button, recoloring a
    specific color depending on hover state.
    """

    def __init__(
        self,
        parent_button,                        # your Button instance
        image: pygame.Surface | str,          # path or loaded surface
        *,
        normal_replace: tuple[int, int, int, int],
        hover_replace: tuple[int, int, int, int],
        size: tuple[int, int] | None = None,  # (width, height) in px
        inset_px: int = 0,                    # optional inset padding
        z_index: int = 10                     # draw order
    ):
        # Fills parent’s area by default
        super().__init__(parent=parent_button,
                         x=0, y=0, width=1.0, height=1.0,
                         x_is_percent=True, y_is_percent=True,
                         width_is_percent=True, height_is_percent=True,
                         z_index=z_index,
                         padding=(inset_px, inset_px, inset_px, inset_px))

        self.mouse_passthrough = True
        self._size = size
        base = _load_surface(image)
        self._img_normal  = _recolor_by_alpha_mask(base, normal_replace)
        self._img_hover   = _recolor_by_alpha_mask(base, hover_replace)
        self._img_disabled = self._make_disabled(self._img_normal)

    def _make_disabled(self, surf: pygame.Surface) -> pygame.Surface:
        """Dim the image for disabled state."""
        out = surf.copy()
        out.fill((255, 255, 255, 153), special_flags=pygame.BLEND_RGBA_MULT)
        return out

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        parent = self.parent
        if not parent.is_enabled:
            img = self._img_disabled
        elif hasattr(parent, "is_hover") and parent.is_hover:
            img = self._img_hover
        else:
            img = self._img_normal

        inner_x, inner_y, inner_w, inner_h = self.get_content_geometry()

        if self._size:
            tw, th = self._size
            blit_img = pygame.transform.smoothscale(img, (tw, th))
        else:
            # default: fit inside content area
            blit_img = pygame.transform.smoothscale(
                img, (inner_w, inner_h)
            )

        bx = inner_x + (inner_w - blit_img.get_width()) // 2
        by = inner_y + (inner_h - blit_img.get_height()) // 2
        surface.blit(blit_img, (bx, by))
