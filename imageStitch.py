from stitching import AffineStitcher
from matplotlib import pyplot as plt
import cv2 as cv


def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

print(AffineStitcher.AFFINE_DEFAULTS)

settings = {# The whole plan should be considered
            "crop": False,
            "detector": "sift",
            # The matches confidences aren't that good
            "confidence_threshold": 0.2}    

stitcher = AffineStitcher(**settings)
panorama = stitcher.stitch(["./input/tree-core3/*.png"])
'''
panorama1 = stitcher.stitch(['input/tree-core_0010.png','input/tree-core_0011.png'])
panorama2 = stitcher.stitch(['input/tree-core_0012.png','input/tree-core2_0013.png'])
settings = {# The whole plan should be considered
            "crop": False,
            "detector": "orb",
            # The matches confidences aren't that good
            "confidence_threshold": 0.2}    
stitcher = AffineStitcher(**settings)
#panorama3 = stitcher.stitch(['input/tree-core_0011.png','input/tree-core_0012.png'])
panorama3 = stitcher.stitch([panorama1, panorama2])
#panorama3 = stitcher.stitch([panorama1, panorama2])
'''
cv.imwrite("output/tree-core3.png", panorama)
plot_image(panorama, (20,20))
