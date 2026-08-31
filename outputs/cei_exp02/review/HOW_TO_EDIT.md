# How to correct a predicted mask in GIMP

Each image here comes as a pair:

* `<stem>_image.png` -- the CEI aerial image. Reference only; never edit it.
* `<stem>_mask.tif`  -- the model's draft label. **This is the file you edit.**

## One-time setup

1. Open GIMP.
2. Load the class colors: **Windows > Dockable Dialogs > Palettes**, right-click
   in the list, **Import Palette...**, choose *Palette file*, and select
   `oem_palette.gpl` from this folder. You now have the 8 class colors plus
   "Unlabeled (ignore)" and can click one to make it the foreground color.
3. Pick the **Pencil** tool (`N`), not the Paintbrush. The pencil has hard edges.
   The paintbrush anti-aliases, which invents blended colors that are not real
   classes. (Those get snapped back to the nearest class on import, so nothing
   breaks -- but the boundary ends up a pixel or two off from what you drew.)

## Editing one mask

1. **File > Open** -> `<stem>_mask.tif`.
2. **File > Open as Layers** -> `<stem>_image.png`. The aerial image lands on top.
3. In the **Layers** panel, drag the image layer **below** the mask layer, then
   lower the *mask* layer's **Opacity** to about 60%. You can now see the aerial
   image through the mask and spot where the model got it wrong.
4. Select the mask layer, pick a class color from the palette, and paint the
   corrections with the Pencil. Zoom in (`+`) for boundaries.
5. Set the mask layer's Opacity back to 100%.
6. **File > Export As...**, keep the name `<stem>_mask.tif`, and overwrite.
   In the TIFF export dialog, LZW compression is fine. Do **not** "Save As" an
   `.xcf` and stop there -- the import step reads the `.tif`.

## Class colors

| Class | Color (RGB) |
| --- | --- |
| Bareland | 128, 0, 0 |
| Rangeland | 0, 255, 36 |
| Developed space | 148, 148, 148 |
| Road | 255, 255, 255 |
| Tree | 34, 97, 38 |
| Water | 0, 69, 255 |
| Agriculture land | 75, 181, 73 |
| Building | 222, 31, 7 |
| Unlabeled (ignore) | 0, 0, 0 |

Paint a region black ("Unlabeled") when you genuinely cannot tell what it is.
Those pixels are excluded from the loss instead of teaching the model a guess.

## When you are done

Run the import step to convert the corrected masks into dataset labels:

    python tools/cei/import_from_gimp.py --review <this folder> --output data/CEI_data/labels
