//
// Created by pbahr on 13/02/2020.
//

#ifndef MOLFLOW_PROJ_POLY2TRICONVERTER_H
#define MOLFLOW_PROJ_POLY2TRICONVERTER_H

#include "ModelReader.h"

class Poly2TriConverter {

    static std::vector<int3> Triangulate(std::vector<float2> &vertices, std::vector<uint32_t> &indices);

public:
    static std::vector<flowgpu::Polygon> PolygonsToTriangles(std::vector<flowgpu::TempFacet>& facets);
    static int
    PolygonsToTriangles(flowgpu::PolygonMesh *polygonMesh, flowgpu::TriangleMesh *triangleMesh);
};


#endif //MOLFLOW_PROJ_POLY2TRICONVERTER_H
