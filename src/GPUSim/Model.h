//
// Created by pbahr on 08/10/2019.
//

#ifndef MOLFLOW_PROJ_MODEL_H
#define MOLFLOW_PROJ_MODEL_H


#pragma once

#include <vector>
//#include "Vector.h"
//#include "Geometry_shared.h"
//#include "Simulation.h"
#include "LaunchParams.h"
#include "../Simulation/MolflowSimGeom.h"
#include "../Parameter.h"
#include <cereal/cereal.hpp>
#include "GPUSettings.h"

/*! \namespace flowgpu - Molflow GPU code */
namespace flowgpu {

    struct Mesh {
        Mesh() : nbFacets(0), nbVertices(0){};
        Mesh(const Mesh&) = delete;

        //std::vector<uint32_t> indices;
        //std::vector<float2> vertices2d;
        std::vector<flowgpu::FacetType> sbtIndices;

        std::vector<float3> vertices3d;
        std::vector<flowgpu::Polygon> poly;

        std::vector<float2> facetProbabilities;
        std::vector<float> cdfs_1; // should not be part of each mesh, but the model itself
        std::vector<float> cdfs_2; // should not be part of each mesh, but the model itself

        uint32_t nbFacets;
        uint32_t nbVertices;
    };

    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh : public Mesh {
        TriangleMesh() : Mesh(){};
        TriangleMesh(const TriangleMesh&) = delete;
        //std::vector<float3> vertex;
        //std::vector<float3> normal;
        //std::vector<int3> index;
        std::vector<int3> indices;
        std::vector<float2> texCoords;

        // material data:
        //float3              diffuse;
    };

    struct PolygonMesh : public Mesh {
        PolygonMesh() : Mesh(){};
        PolygonMesh(const PolygonMesh&) = delete;

        std::vector<uint32_t> indices;
        std::vector<float2> vertices2d;
        std::vector<double2> vertices2d64;

    };

    //TODO: Unify with buffer_shared.h
    /*struct GeomProperties {  //Formerly SHGEOM
        size_t     nbFacet;   // Number of facets (total)
        size_t     nbVertex;  // Number of 3D vertices
        size_t     nbSuper;   // Number of superstructures
        std::string name;  // (Short file name)

        template <class Archive> void serialize(Archive & archive) {
            archive(
                    CEREAL_NVP(nbFacet),   // Number of facets (total)
                    CEREAL_NVP(nbVertex),  // Number of 3D vertices
                    CEREAL_NVP(nbSuper),   // Number of superstructures
                    CEREAL_NVP(name)  // (Short file name)
            );
        }
    };*/

    struct Model {
        Model() : ontheflyParams{}, nbFacets_total(), nbVertices_total(), nbTexel_total{}, nbProfSlices_total{}{};
        Model(const Model&) = delete;
        ~Model()
        {
            //for (auto mesh : meshes) delete mesh;
            for (auto& mesh : poly_meshes) {
                delete mesh;
            }
            for (auto& mesh : triangle_meshes) {
                delete mesh;
            }
        }

        void Reset(){
            triangle_meshes.clear();
            poly_meshes.clear();
            tri_facetOffset.clear();
            textures.clear();
            facetTex.clear();
            texInc.clear();
            cdfs_1.clear();
            cdfs_2.clear();
            profiles.clear();
        }
        std::vector<TriangleMesh *> triangle_meshes;
        std::vector<PolygonMesh *> poly_meshes;
        std::vector<size_t> tri_facetOffset;

        std::vector<flowgpu::Texel> textures;
        std::vector<flowgpu::FacetTexture> facetTex;
        std::vector<float> texInc;
        std::vector<float> cdfs_1;
        std::vector<float> cdfs_2;

        std::vector<flowgpu::Texel> profiles;

        uint32_t nbFacets_total;
        uint32_t nbVertices_total;
        uint32_t nbTexel_total;
        uint32_t nbProfSlices_total;

#if defined(USE_NORMALIZED_GEOM)
        double maxAbsCoord = 0.0; // scaling factor for normalization
#endif
        //! bounding box of all vertices in the model
        //float3 bounds.lower;
        //float3 bounds.upper;

        // Global Settings
        // Should they be here as well or only LaunchParams?

        //MolflowGPUSettings parametersGlobal;
        GeomProperties geomProperties;

        // Molflow structures
        // Geometry
        WorkerParams wp;
        //GeomProperties sh;
        OntheflySimulationParams ontheflyParams;

        std::vector<SuperStructure> structures; //They contain the facets
        std::vector<std::vector<std::pair<double, double>>> CDFs; //cumulative distribution function for each temperature
        std::vector<std::vector<std::pair<double, double>>> IDs; //integrated distribution function for each time-dependent desorption type
        std::vector<double> temperatures; //keeping track of all temperatures that have a CDF already generated
        std::vector<double> moments;      //time values (seconds) when a simulation state is measured
        std::vector<size_t> desorptionParameterIDs; //time-dependent parameters which are used as desorptions, therefore need to be integrated
        std::vector<Parameter> parameters; //Time-dependent parameters
    };

    //Model *loadOBJ(const std::string &objFile);
    //Model *loadFromMolflow(const std::vector<Vector3d> &geomVertices, const std::vector<SuperStructure> &structures, const WorkerParams& wp, const std::vector<std::vector<std::pair<double,double>>> CDFs);
}



#endif //MOLFLOW_PROJ_MODEL_H
