"""
This module contains a zero-order representation of an imaginary separator for RO
used to separate specific solute stream from tds stream which is not perfectly removed by RO.
"""

from idaes.core import declare_process_block_class
from watertap.core import build_sido, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("ImaginarySeparatorZO")
class ImaginarySeparatorZOData(ZeroOrderBaseData):
    """
    Zero-Order model for an Imaginary Separator for RO.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    def build(self):
        super().build()

        self._tech_type = "imaginary_separator"

        build_sido(self)