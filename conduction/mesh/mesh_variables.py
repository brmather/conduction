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

class MeshVariable(object):
    """
    Mesh variables live on the global mesh
    Every time its data is called a local instance is returned
    """
    def __init__(self, name, dm):
        self._dm = dm
        name = str(name)

        # mesh variable vector
        self._gdata = dm.createGlobalVector()
        self._ldata = dm.createLocalVector()

        self._gdata.setName(name)
        self._ldata.setName(name)

        self.size = self._ldata.getSizes()[0]

    def __delete__(self):
        self._ldata.destroy()
        self._gdata.destroy()

    def __getitem__(self, pos):
        self._dm.globalToLocal(self._gdata, self._ldata)
        return self._ldata[pos]

    def __setitem__(self, pos, value):
        self._ldata[pos] = value
        self._dm.localToGlobal(self._ldata, self._gdata)


    @property
    def array(self):
        self._dm.globalToLocal(self._gdata, self._ldata)
        return self._ldata


    @property
    def data(self):
        pass

    @data.getter
    def data(self):
        self._dm.globalToLocal(self._gdata, self._ldata)
        return self._ldata

    @data.setter
    def data(self, val):
        if type(val) is float:
            self._ldata.set(val)
            self._gdata.set(val)
        else:
            self._ldata.setArray(val)
            self._dm.localToGlobal(self._ldata, self._gdata)

    @data.deleter
    def data(self):
        self._ldata.destroy()
        self._gdata.destroy()

    def getGlobal(self):
        return self._gdata

    def getLocal(self):
        return self._ldata