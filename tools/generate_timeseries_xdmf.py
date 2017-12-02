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

def generateTimeseriesXdmf(*args):
    """
    Generate a XDMF file to visualise HDF5 fields and vectors in Paraview.

    Supply an ordered list of HDF5 filenames to create a timeseries to write
    a XDMF file to the working directory.
    """

    import h5py
    import os

    # We only need the topology information from first HDF5 file
    HDF5_filename = args[0]

    filename = HDF5_filename
    if HDF5_filename.endswith('.h5'):
        filename = HDF5_filename[:-3]
    else:
        HDF5_filename += '.h5'

    filename += '.xmf'

    f = open(filename, 'w')

    def write_header(f):
        f.write('''<?xml version="1.0" ?>\n\
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n\
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n\
  <Information Name="SampleLocation" Value="4"/>\n\
  <Domain>\n''')

    def write_footer(f):
        f.write('''    </Grid>\n  </Domain>\n</Xdmf>''')

    def write_topology(f, dim, shape, origin, stride):
        f.write('''    <Topology TopologyType="3DCORECTMesh" NumberOfElements="{1}"/>\n\
    <Geometry GeometryType="ORIGIN_DXDYDZ">\n\
      <DataItem Dimensions="{0}" NumberType="Float" Precision="4" Format="XML">\n\
        {2}\n\
      </DataItem>\n\
      <DataItem Dimensions="{0}" NumberType="Float" Precision="4" Format="XML">\n\
        {3}\n\
      </DataItem>\n\
    </Geometry>\n'''.format(dim, shape, origin, stride))

    def write_timeseries(f, dim, trange):
        f.write('''    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n\
      <Time TimeType="HyperSlab">\n\
        <DataItem Format="XML" NumberType="Float" Dimensions="{0}">\n\
          {1}\n\
        </DataItem>\n\
      </Time>\n'''.format(dim, trange))

    def write_grid_header(f, step):
        f.write('''      <Grid Name="t{}" GridType="Uniform">\n\
        <Topology Reference="/Xdmf/Domain/Topology[1]"/>\n\
        <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>\n'''.format(step))

    def write_grid_footer(f):
        f.write('''      </Grid>\n''')

    def write_attribute(f, attributeName, arrtype, dshape, dpath):
        f.write('''          <Attribute Name="{0}" AttributeType="{1}" Center="Node">'\n\
            <DataItem Dimensions="{2}" NumberType="Float" Precision="4" Format="HDF">{3}</DataItem>\n\
          </Attribute>\n'''.format(attributeName, arrtype, dshape, dpath))

    def array_to_string(array):
        s = ""
        for i in array:
            s += "{} ".format(i)
        return s


    h5file = h5py.File(HDF5_filename, 'r')
    basename = os.path.basename(HDF5_filename)

    # get topology attributes
    topo = h5file['topology']
    minCoords = topo.attrs['minCoord']
    maxCoords = topo.attrs['maxCoord']
    shape = topo.attrs['shape']
    dim = len(shape)
    nsteps = len(args)

    nodes = 1
    for n in shape:
        nodes *= n

    stride = (maxCoords - minCoords)/shape

    tshape = array_to_string(shape)
    torigin = array_to_string(minCoords)
    tstride = array_to_string(stride)
    trange = array_to_string(list(range(0, nsteps)))

    write_header(f)
    write_topology(f, dim, tshape, torigin, tstride)
    write_timeseries(f, dim, trange)

    for step, HDF5_filename in enumerate(args):
        h5file = h5py.File(HDF5_filename, 'r')
        basename = os.path.basename(HDF5_filename)

        write_grid_header(f, step)
        for key in h5file:
            dset = h5file[key]

            # We only want datasets, topology group is not required
            if type(dset) is h5py.Dataset:

                dpath = basename + ":" + dset.name
                dname = dset.name[1:]
                dshape = array_to_string(dset.shape)

                dnodes = 1
                for n in dset.shape:
                    dnodes *= n

                if dnodes%nodes != 0:
                    raise ValueError("Dataset {} is not a valid shape".format(dname))
                elif dnodes/nodes > 1:
                    arrtype = "Vector"
                elif dnodes/nodes == 1:
                    arrtype = "Scalar"

                write_attribute(f, dname, arrtype, dshape, dpath)
        write_grid_footer(f)
        h5file.close()

    write_footer(f)
    f.close()

if __name__ == '__main__':
    import sys
    generateTimeseriesXdmf(sys.argv[1:])