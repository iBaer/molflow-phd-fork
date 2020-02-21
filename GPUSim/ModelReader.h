//
// Created by pbahr on 12/02/2020.
//

#ifndef MOLFLOW_PROJ_MODELREADER_H
#define MOLFLOW_PROJ_MODELREADER_H

#include "Model.h"

/*! \namespace flowgeom - Molflow Geometry code */
namespace flowgeom {
    /*int loadFromMolflow(
                Geometry& geometry, const WorkerParams& wp, const std::vector<std::vector<std::pair<double,double>>> CDFs);*/
    /*flowgpu::Model *loadFromMolflowSimu(
            const std::vector<Vector3d> &geomVertices, const std::vector<SuperStructure> &structures,
            const WorkerParams& wp, const std::vector<std::vector<std::pair<double,double>>> CDFs);*/

    // for loading from a Molflow/cereal export
    struct TempFacet{
        FacetProperties facetProperties;
        std::vector<uint32_t> indices;
        std::vector<float2> vertices2;

        template<class Archive>
        void serialize(Archive & archive) {
            archive(
                    facetProperties,
                    indices,
                    vertices2
            );
        }
    };

    flowgpu::Model* initializeModel(std::string fileName);
    flowgpu::Model* loadFromSerialization(std::string fileName);
}


#endif //MOLFLOW_PROJ_MODELREADER_H
