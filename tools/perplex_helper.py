"""
Copyright 2017 Ben Mather

This file is part of Conduction <https://git.dias.ie/itherc/conduction/>

Conduction is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Conduction is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Conduction.  If not, see <http://www.gnu.org/licenses/>.
"""

class VelocityTable(object):

    def __init__(self, coords):
        from scipy.spatial import cKDTree
        self.tree = cKDTree(coords)

    def add_table(self, velocity_field, index):

        self.vdict[index] = velocity_field

    def __call__(self, xi, index):

        d, idx = self.tree.query(xi)
        return self.vdict[index][idx]
