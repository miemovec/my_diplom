import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import re
import os

class DXF2IMG:
    default_img_format = '.png'
    default_img_res = 300
    default_bg_color = '#FFFFFF'

    def convert_dxf2img(self, name, img_format=default_img_format, img_res=default_img_res, clr=default_bg_color):
        try:
            doc = ezdxf.readfile(name)
            msp = doc.modelspace()

            auditor = doc.audit()
            if len(auditor.errors) != 0:
                raise Exception("This DXF document is damaged and can't be converted!", name)

            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            ctx.set_current_layout(msp)
            ezdxf.addons.drawing.properties.MODEL_SPACE_BG_COLOR = clr
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)

            img_name = os.path.splitext(os.path.basename(name))[0] + img_format
            fig.savefig(img_name, dpi=img_res)
            print(f"{name} -> {img_name} converted successfully")

        except Exception as e:
            print(f"Error converting {name}: {e}")

if __name__ == '__main__':
    file_path = '/data/dxf/example1.dxf'
    converter = DXF2IMG()
    converter.convert_dxf2img(file_path)
