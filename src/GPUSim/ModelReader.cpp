/*
Program:     MolFlow+ / Synrad+
Description: Monte Carlo simulator for ultra-high vacuum and synchrotron radiation
Authors:     Jean-Luc PONS / Roberto KERSEVAN / Marton ADY / Pascal BAEHR
Copyright:   E.S.R.F / CERN
Website:     https://cern.ch/molflow

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Full license text: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
*/

#include "ModelReader.h"
#include "Poly2TriConverter.h"
//#include "Facet_shared.h"

// debug output
#include <fstream>

#include "helper_math.h"
#include "../MolflowTypes.h"
//#include "Serializations.h"
#include "Serializations.h"
#include "OptixPolygon.h"
#include "fmt/core.h"
#include "NeighborScan.h"
#include "Helper/MathTools.h"
#include "../Simulation/MolflowSimFacet.h"
#include "Helper/Chronometer.h"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

/*#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>*/

inline void vector3d_to_float3(float3 &t, const Vector3d &o) {
    t.x = o.x;
    t.y = o.y;
    t.z = o.z;

    return;
}

inline float3 vector3d_to_float3(const Vector3d &o) {
    float3 t;
    t.x = o.x;
    t.y = o.y;
    t.z = o.z;

    return t;
}

inline void vector3d_to_double3(double3 &t, const Vector3d &o) {
    t.x = o.x;
    t.y = o.y;
    t.z = o.z;

    return;
}

inline double3 vector3d_to_double3(const Vector3d &o) {
    double3 t;
    t.x = o.x;
    t.y = o.y;
    t.z = o.z;

    return t;
}
/*! \namespace flowgeom - Molflow Geometry code */
namespace flowgpu {

    // split a 2sided facet into two 1sided (e.g. in case of culling)
    void split2sided(flowgpu::TriangleMesh *triMesh) {
        int facetIndex = 0;
        const int initial_size = triMesh->poly.size();
        for (int fid = 0; fid < initial_size; fid++) {
            auto &facet = triMesh->poly[fid];
            //for (auto &facet: triMesh->poly) {
            if (facet.facProps.is2sided) {
                auto poly_cpy = facet;
                // Calculate triangle area
                auto &triIndices = triMesh->indices[facetIndex];
                auto &a = triMesh->vertices3d[triIndices.x];
                auto &b = triMesh->vertices3d[triIndices.y];
                auto &c = triMesh->vertices3d[triIndices.z];

                auto indices_reverse = triIndices;
                std::swap(triIndices.x, triIndices.z);

                // clone vertices, indices, facets
                //poly_cpy.
                triMesh->indices.emplace_back(indices_reverse);
                poly_cpy.N *= -1.0;
                poly_cpy.facProps.is2sided = false;
                facet.facProps.is2sided = false;
                triMesh->poly.emplace_back(poly_cpy);

                triMesh->nbFacets++;
            }
            ++facetIndex;
        }
    }

    void convertFacet2Poly(const std::vector<TempFacet> &facets, std::vector<flowgpu::Polygon> &convertedPolygons) {

        int32_t vertCount = 0;
        std::vector<size_t> deleteFacets; //mark some facets for deleting e.g. outgassing on linked

        for (int i = 0; i < facets.size(); ++i) {
            auto &temp = facets[i];

            flowgpu::Polygon polygon(temp.vertices2.size());
            polygon.facProps.stickingFactor = temp.facetProperties.sticking;
            polygon.facProps.temperature = temp.facetProperties.temperature;

            //TODO: For now, treat 'link facets' like transparent facets
            if (temp.facetProperties.superDest) {
                polygon.facProps.opacity = 0.0f;
            } else {
                polygon.facProps.opacity = temp.facetProperties.opacity;
            }
            polygon.facProps.is2sided = temp.facetProperties.is2sided;
            polygon.facProps.cdf_id = temp.facetProperties.CDFid;

            // desorption
            polygon.desProps.desorbType = temp.facetProperties.desorbType;
            polygon.desProps.cosineExponent = temp.facetProperties.desorbTypeN;

            if (polygon.nbVertices != temp.facetProperties.nbIndex) {
                polygon.nbVertices = temp.facetProperties.nbIndex;
                std::cout << "Parsing error! Vert size != nbIndex" << std::endl;
                exit(0);
            }

            polygon.indexOffset = vertCount;
            vector3d_to_float3(polygon.O, temp.facetProperties.O);
            vector3d_to_float3(polygon.U, temp.facetProperties.U);
            vector3d_to_float3(polygon.V, temp.facetProperties.V);
            vector3d_to_float3(polygon.Nuv, temp.facetProperties.Nuv);
            vector3d_to_float3(polygon.nU, temp.facetProperties.nU);
            vector3d_to_float3(polygon.nV, temp.facetProperties.nV);
            vector3d_to_float3(polygon.N, temp.facetProperties.N);

            vector3d_to_double3(polygon.Ox64, temp.facetProperties.O);
            vector3d_to_double3(polygon.Ux64, temp.facetProperties.U);
            vector3d_to_double3(polygon.Vx64, temp.facetProperties.V);
            vector3d_to_double3(polygon.Nuvx64, temp.facetProperties.Nuv);
            vector3d_to_double3(polygon.nUx64, temp.facetProperties.U);
            vector3d_to_double3(polygon.nVx64, temp.facetProperties.V);
            vector3d_to_double3(polygon.Nx64, temp.facetProperties.N);

            /*polygon.O = temp.facetProperties.O;
            polygon.U = temp.facetProperties.U;
            polygon.V = temp.facetProperties.V;
            polygon.Nuv = temp.facetProperties.Nuv;
            polygon.nU = temp.facetProperties.nU;
            polygon.nV = temp.facetProperties.nV;
            polygon.N = temp.facetProperties.N;*/

            polygon.parentIndex = i;
            vertCount += polygon.nbVertices;

            convertedPolygons.push_back(std::move(polygon));

        }

    }

    //! Calculate outgassing values in relation to (tri_area / poly_area)
    void CalculateRelativePolygonOutgassing(const std::vector<TempFacet> &facets, flowgpu::PolygonMesh *polyMesh) {
        float fullOutgassing = 0;
        int facetIndex = 0;
        for (auto &facet: polyMesh->poly) {

            // outgassing of a triangle is only a percentage of the original polygon's

            float fullOutgassing_inc = fullOutgassing +
                                       (facets[facet.parentIndex].facetProperties.outgassing) /
                                       (1.38E-23 * facets[facet.parentIndex].facetProperties.temperature);
            polyMesh->facetProbabilities.push_back(make_float2(fullOutgassing, fullOutgassing_inc));
            fullOutgassing = fullOutgassing_inc;

            ++facetIndex;
        }
        for (auto &facetProb: polyMesh->facetProbabilities) {
            facetProb.x /= fullOutgassing; // normalize to [0,1]
            facetProb.y /= fullOutgassing; // normalize to [0,1]
        }
    };


    //! Calculate triangle center for later offsetting
    void CalculateTriangleCenter(flowgpu::TriangleMesh *triMesh) {
        float fullOutgassing = 0;
        int facetIndex = 0;
        for (auto &facet: triMesh->poly) {
            // Calculate triangle center
            auto &triIndices = triMesh->indices[facetIndex];
            auto &a = triMesh->vertices3d[triIndices.x];
            auto &b = triMesh->vertices3d[triIndices.y];
            auto &c = triMesh->vertices3d[triIndices.z];

            facet.center = (a + b + c) / 3.0;
            //facet.center += facet.N * 1e-2f;

            Log::console_msg(5, "Facet {}({}) with center {} , {} , {} and N = {} , {} , {}\n",
                             facetIndex, facet.parentIndex, facet.center.x, facet.center.y, facet.center.z,
                             facet.N.x, facet.N.y, facet.N.z);
            Log::console_msg(5, "     {} , {} , {} -- {} , {} , {} -- {} , {} , {}\n",
                             a.x, a.y, a.z,
                             b.x, b.y, b.z,
                             c.x, c.y, c.z
            );
            ++facetIndex;
        }
    };

    //! Calculate outgassing values in relation to (tri_area / poly_area)
    void CalculateRelativeTriangleOutgassing(const std::vector<TempFacet> &facets, flowgpu::TriangleMesh *triMesh) {
        float fullOutgassing = 0;
        int facetIndex = 0;
        for (auto &facet: triMesh->poly) {

            // Calculate triangle area
            auto &triIndices = triMesh->indices[facetIndex];
            auto &a = triMesh->vertices3d[triIndices.x];
            auto &b = triMesh->vertices3d[triIndices.y];
            auto &c = triMesh->vertices3d[triIndices.z];


            float3 ab = make_float3((b.x - a.x), (b.y - a.y), (b.z - a.z));
            float3 ac = make_float3((c.x - a.x), (c.y - a.y), (c.z - a.z));
            float area = 0.5 * length(cross(ab, ac));

            // outgassing of a triangle is only a percentage of the original polygon's
            float areaPercentageOfPoly = (area / facets[facet.parentIndex].facetProperties.area);

            float fullOutgassing_inc = fullOutgassing +
                                       (facets[facet.parentIndex].facetProperties.outgassing * areaPercentageOfPoly) /
                                       (1.38E-23 * facets[facet.parentIndex].facetProperties.temperature);
            triMesh->facetProbabilities.push_back(make_float2(fullOutgassing, fullOutgassing_inc));
            fullOutgassing = fullOutgassing_inc;

            ++facetIndex;
        }
        for (auto &facetProb: triMesh->facetProbabilities) {
            facetProb.x /= fullOutgassing; // normalize to [0,1]
            facetProb.y /= fullOutgassing; // normalize to [0,1]
        }
    };

    int InitializeTexture(TempFacet &facet,
                          std::vector<Texel> &texture,
                          FacetTexture &facetTex,
                          std::vector<float> &texInc) {

        //Textures
        if (facet.facetProperties.isTextured) {
            size_t nbE = facet.facetProperties.texWidth * facet.facetProperties.texHeight;
            facetTex.texHeight = facet.facetProperties.texHeight;
            facetTex.texWidth = facet.facetProperties.texWidth;
            facetTex.texHeight_precise = facet.facetProperties.texHeight_precise;
            facetTex.texWidth_precise = facet.facetProperties.texWidth_precise;

            // Add a cutoff to allow for texture positions [0.0,1.0], opposed to [0.0,1.0[
            const float cutOff = 0.9999999f;
            facetTex.texHeight_precise *= cutOff;
            facetTex.texWidth_precise *= cutOff;

            if (facet.texelInc.size() != nbE) {
                printf("texture inc vector has weird size: %zu should be %zu\n", facet.texelInc.size(), nbE);
                return 0;
            }

            try {
                texture = std::vector<Texel>(nbE);
                for (auto it = facet.texelInc.begin(); it != facet.texelInc.end(); it++)
                    texInc.emplace_back(*it);
                //texInc = facet.texelInc;
            }
            catch (...) {
                printf("Not enough memory to load textures\n");
                return 0;
            }
        }
        return 1;
    }

    int InitializeProfile(TempFacet &facet, std::vector<Texel> &profile) {

        //Profile
        if (facet.facetProperties.isProfile) {
            //const size_t PROFILE_SIZE = 100;

            try {
                // PROFILE_SIZE bins per profile
                profile.resize(PROFILE_SIZE);
            }
            catch (...) {
                printf("Not enough memory to load profiles\n");
                return 0;
            }
        }
        return 1;
    }

    int parseGeomFromSerialization(flowgpu::Model *model, std::vector<flowgpu::TempFacet> &facets,
                                   std::vector<float3> &vertices3d) {
        // First create a regular polygonmesh
        // transform Molflow facet data to simulation polygons
        // transform polygons to triangles
        flowgpu::PolygonMesh *polyMesh = new flowgpu::PolygonMesh();
        convertFacet2Poly(facets, polyMesh->poly);

        int i = 0;
        int indexOffset = 0;

        for (auto &facet: facets) {
            polyMesh->poly[i++].indexOffset = indexOffset;

            assert(facet.indices.size() > 0);
            indexOffset += facet.indices.size();
            for (auto ind: facet.indices) {
                polyMesh->indices.push_back(ind);
            }
            for (auto vert: facet.vertices2) {
                polyMesh->vertices2d.push_back(make_float2(vert.u, vert.v));
                polyMesh->vertices2d64.push_back(make_double2(vert.u, vert.v));

            }
        }
        polyMesh->vertices3d = vertices3d;

        polyMesh->nbFacets = model->geomProperties.nbFacet;
        polyMesh->nbVertices = model->geomProperties.nbVertex;

#ifdef WITHTRIANGLES
        // Now create Triangle Mesh
        {
            auto *triMesh = new flowgpu::TriangleMesh();
            try {
                Poly2TriConverter::PolygonsToTriangles(polyMesh, triMesh);
            }
            catch (std::exception &e) {
                std::cerr << e.what() << std::endl;
                return -1;
            }
            triMesh->vertices3d = polyMesh->vertices3d;
            triMesh->nbVertices = triMesh->poly.size() * 3;
            triMesh->nbFacets = triMesh->poly.size();

            //triMesh->cdfs_1.push_back(0);

            split2sided(triMesh);


            CalculateTriangleCenter(triMesh);
            //--- Calculate outgassing values in relation to (tri_area / poly_area)
            CalculateRelativeTriangleOutgassing(facets, triMesh);

            model->triangle_meshes.push_back(triMesh);
        }

#else
        CalculateRelativePolygonOutgassing(facets, polyMesh);
#endif

        if (!polyMesh->poly.empty()) {
            //polyMesh->cdfs_1.push_back(0);
            model->poly_meshes.push_back(polyMesh);
        } else {
            delete polyMesh;
        }
        for (auto &pm: model->poly_meshes) {
            model->nbFacets_total += pm->nbFacets;
            model->nbVertices_total += pm->nbVertices;
        }
        for (auto &triMesh: model->triangle_meshes) {
            model->nbFacets_total += triMesh->nbFacets;
            model->nbVertices_total += triMesh->nbVertices;
        }
        model->geomProperties.nbFacet = model->nbFacets_total;
        model->geomProperties.nbVertex = model->nbVertices_total;

        //TODO: Remove
        int hitOffset = 0;
        for (auto &facet: facets) {
            model->tri_facetOffset.emplace_back(hitOffset);
        }

        for (auto &mesh: model->poly_meshes) {
            std::vector<flowgpu::FacetType> sbtIndices; // Facet Type
            for (auto &polygon: mesh->poly) {
                //if(polygon.facProps.is2sided && polygon.facProps.opacity == 0.0f){
                if (polygon.facProps.opacity == 0.0f) {
                    //std::cout << sbtIndices.size() << " > is transparent " << std::endl;
                    sbtIndices.emplace_back(FacetType::FACET_TYPE_TRANS);
                } else {
                    sbtIndices.emplace_back(FacetType::FACET_TYPE_SOLID);
                }
            }
            mesh->sbtIndices = sbtIndices;
        }

        size_t n_transparent = 0;
        for (auto &mesh: model->triangle_meshes) {
            std::vector<flowgpu::FacetType> sbtIndices; // Facet Type
            sbtIndices.reserve(mesh->poly.size());
            for (auto &triangle: mesh->poly) {
                //if(triangle.facProps.is2sided && triangle.facProps.opacity == 0.0f){
                if (triangle.facProps.opacity == 0.0f) {
                    n_transparent++;//std::cout << sbtIndices.size() << " > is transparent " << std::endl;
                    sbtIndices.emplace_back(FacetType::FACET_TYPE_TRANS);
                } else {
                    sbtIndices.emplace_back(FacetType::FACET_TYPE_SOLID);
                }
            }
            mesh->sbtIndices = sbtIndices;
        }

        if (n_transparent)
            Log::console_msg(4, "Model has {} transparent facets.\n");
        // Textures
        if (!model->textures.empty()) {
            std::cout << "[WARNING] Textures would get added to non-empty vector!" << std::endl;
            return 1;
        }

        int textureOffset = 0;
        int texelOffset = 0;
        for (int facetInd = 0; facetInd < facets.size(); ++facetInd) {
            auto &facet = facets[facetInd];
            if (facet.facetProperties.isTextured) {

                std::vector<Texel> texture;
                std::vector<float> texInc;
                FacetTexture facetTex; // TODO: Map Offset to Triangle
                if (!InitializeTexture(facet, texture, facetTex, texInc)) {
                    printf("[ERROR] Initializing facetTex #%d\n", facetInd);
                    exit(0);
                }

                facetTex.bbMin = make_float3(std::numeric_limits<float>::max());
                facetTex.bbMax = make_float3(std::numeric_limits<float>::lowest());

                for (auto &ind: facet.indices) {
                    facetTex.bbMin = fminf(facetTex.bbMin, vertices3d[ind]);
                    facetTex.bbMax = fmaxf(facetTex.bbMax, vertices3d[ind]);
                }

                Log::console_msg(42, "Texture BBox: {} , {} , {} | {} , {} , {}\n",
                                 facetTex.bbMin.x, facetTex.bbMin.y, facetTex.bbMin.z,
                                 facetTex.bbMax.x, facetTex.bbMax.y, facetTex.bbMax.z);


                facetTex.texelOffset = texelOffset;
                texelOffset += texture.size();

                model->textures.insert(std::end(model->textures), std::begin(texture), std::end(texture));
                model->texInc.insert(std::end(model->texInc), std::begin(texInc), std::end(texInc));
                model->facetTex.push_back(facetTex);

                for (auto &polyMesh: model->poly_meshes) {
                    for (auto &polygon: polyMesh->poly) {
                        if (polygon.parentIndex == facetInd) {
                            polygon.texProps.textureOffset = textureOffset;
                            polygon.texProps.textureSize = texture.size();
                            if (facet.facetProperties.countAbs)
                                polygon.texProps.textureFlags |= TEXTURE_FLAGS::countAbs;
                            if (facet.facetProperties.countRefl)
                                polygon.texProps.textureFlags |= TEXTURE_FLAGS::countRefl;
                            if (facet.facetProperties.countTrans)
                                polygon.texProps.textureFlags |= TEXTURE_FLAGS::countTrans;
                            if (facet.facetProperties.countDirection)
                                polygon.texProps.textureFlags |= TEXTURE_FLAGS::countDirection;
                            if (facet.facetProperties.countDes)
                                polygon.texProps.textureFlags |= TEXTURE_FLAGS::countDes;
                        }
                    }
                }
                for (auto &triMesh: model->triangle_meshes) {
                    uint32_t triInd = 0;
                    for (auto &triangle: triMesh->poly) {
                        if (triangle.parentIndex == facetInd) {
                            triangle.texProps.textureOffset = textureOffset;
                            triangle.texProps.textureSize = texture.size();
                            if (facet.facetProperties.countAbs)
                                triangle.texProps.textureFlags |= TEXTURE_FLAGS::countAbs;
                            if (facet.facetProperties.countRefl)
                                triangle.texProps.textureFlags |= TEXTURE_FLAGS::countRefl;
                            if (facet.facetProperties.countTrans)
                                triangle.texProps.textureFlags |= TEXTURE_FLAGS::countTrans;
                            if (facet.facetProperties.countDirection)
                                triangle.texProps.textureFlags |= TEXTURE_FLAGS::countDirection;
                            if (facet.facetProperties.countDes)
                                triangle.texProps.textureFlags |= TEXTURE_FLAGS::countDes;

                            if (triangle.texProps.textureFlags != noTexture) {
                                // TODO: Map Texcoords

                                /*std::cout << triangle.parentIndex << " | " << triInd << " => ";
                                std::cout << " Origin " << triangle.O.x << " , " << triangle.O.y << " , " << triangle.O.z << " => ";
                                std::cout << " U " << triangle.U.x << " , " << triangle.U.y << " , " << triangle.U.z << " => ";
                                std::cout << " V " << triangle.V.x << " , " << triangle.V.y << " , " << triangle.V.z << " => ";
                                //facetTex. = destination->sh.O + u * destination->sh.U + v * destination->sh.V;
                                float3 c1 = (triMesh->vertices3d[triMesh->indices[triInd].x] - facetTex.bbMin) / (facetTex.bbMax-facetTex.bbMin);
                                float3 c2 = (triMesh->vertices3d[triMesh->indices[triInd].y] - facetTex.bbMin) / (facetTex.bbMax-facetTex.bbMin);
                                float3 c3 = (triMesh->vertices3d[triMesh->indices[triInd].z] - facetTex.bbMin) / (facetTex.bbMax-facetTex.bbMin);

                                std::cout << c1.x << " , " << c1.y << " , " << c1.z << " | ";
                                std::cout << c2.x << " , " << c2.y << " , " << c2.z << " | ";
                                std::cout << c3.x << " , " << c3.y << " , " << c3.z << " | ";
                                std::cout << "---" << std::endl;*/
                            }

                        }

                        triInd++;
                    }
                }
                textureOffset += 1;
            } // one more texture
        }

        Log::console_msg(2, "Preparing texture coords for all triangles\n");
        // tex coords for all triangles
        {

            for (auto &triMesh: model->triangle_meshes) {
                triMesh->texCoords.reserve(textureOffset);


                for (int facetInd = 0; facetInd < facets.size(); ++facetInd) {
                    auto &facet = facets[facetInd];
                    uint32_t triInd = 0;
                    for (auto &triangle: triMesh->poly) {
                        if (triangle.parentIndex == facetInd) {
                            // TODO: Map Texcoords

                            // texture coords for all triangles even without textures
                            double u = 0.0;
                            double v = 0.0;
                            if (facet.facetProperties.V.x * facet.facetProperties.U.y != 0.0) {
                                double det = facet.facetProperties.U.x * facet.facetProperties.V.y -
                                             facet.facetProperties.U.y *
                                             facet.facetProperties.V.x; // TODO: Pre calculate

                                // Vert 1
                                double3 b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].x]) -
                                            vector3d_to_double3(facet.facetProperties.O);
                                double detU = b.x * facet.facetProperties.V.y - b.y * facet.facetProperties.V.x;
                                double detV = facet.facetProperties.U.x * b.y - facet.facetProperties.U.y * b.x;

                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));

                                // Vert 2
                                b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].y]) -
                                    vector3d_to_double3(facet.facetProperties.O);
                                detU = b.x * facet.facetProperties.V.y - b.y * facet.facetProperties.V.x;
                                detV = facet.facetProperties.U.x * b.y - facet.facetProperties.U.y * b.x;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));

                                // Vert 3
                                b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].z]) -
                                    vector3d_to_double3(facet.facetProperties.O);
                                detU = b.x * facet.facetProperties.V.y - b.y * facet.facetProperties.V.x;
                                detV = facet.facetProperties.U.x * b.y - facet.facetProperties.U.y * b.x;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));
                            } else if (facet.facetProperties.V.y * facet.facetProperties.U.z != 0.0) {
                                double det = facet.facetProperties.U.y * facet.facetProperties.V.z -
                                             facet.facetProperties.U.z *
                                             facet.facetProperties.V.y; // TODO: Pre calculate

                                // Vert 1
                                double3 b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].x]) -
                                            vector3d_to_double3(facet.facetProperties.O);
                                double detU = b.y * facet.facetProperties.V.z - b.z * facet.facetProperties.V.y;
                                double detV = facet.facetProperties.U.y * b.z - facet.facetProperties.U.z * b.y;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));

                                // Vert 2
                                b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].y]) -
                                    vector3d_to_double3(facet.facetProperties.O);
                                detU = b.y * facet.facetProperties.V.z - b.z * facet.facetProperties.V.y;
                                detV = facet.facetProperties.U.y * b.z - facet.facetProperties.U.z * b.y;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));

                                // Vert 3
                                b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].z]) -
                                    vector3d_to_double3(facet.facetProperties.O);
                                detU = b.y * facet.facetProperties.V.z - b.z * facet.facetProperties.V.y;
                                detV = facet.facetProperties.U.y * b.z - facet.facetProperties.U.z * b.y;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));
                            } else if (facet.facetProperties.V.z * facet.facetProperties.U.x != 0.0) {
                                double det = facet.facetProperties.U.z * facet.facetProperties.V.x -
                                             facet.facetProperties.U.x *
                                             facet.facetProperties.V.z; // TODO: Pre calculate

                                // Vert 1
                                double3 b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].x]) -
                                            vector3d_to_double3(facet.facetProperties.O);
                                double detU = b.z * facet.facetProperties.V.x - b.x * facet.facetProperties.V.z;
                                double detV = facet.facetProperties.U.z * b.x - facet.facetProperties.U.x * b.z;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));

                                // Vert 2
                                b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].y]) -
                                    vector3d_to_double3(facet.facetProperties.O);
                                detU = b.z * facet.facetProperties.V.x - b.x * facet.facetProperties.V.z;
                                detV = facet.facetProperties.U.z * b.x - facet.facetProperties.U.x * b.z;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));

                                // Vert 3
                                b = make_double3(triMesh->vertices3d[triMesh->indices[triInd].z]) -
                                    vector3d_to_double3(facet.facetProperties.O);
                                detU = b.z * facet.facetProperties.V.x - b.x * facet.facetProperties.V.z;
                                detV = facet.facetProperties.U.z * b.x - facet.facetProperties.U.x * b.z;
                                v = (detV) / (det);
                                u = (detU) / det;
                                triMesh->texCoords.emplace_back(
                                        make_float2(static_cast<float>(u), static_cast<float>(v)));
                            }

                            //TODO: Check if legit
                            break;
                        }

                        triInd++;
                    }
                }
            }
        }

#ifdef BOUND_CHECK
        model->nbTexel_total = texelOffset;
#endif

        Log::console_msg(2, "Preparing profiles for all triangles\n");
        // Profiles
        if (!model->profiles.empty()) {
            std::cout << "[WARNING] Profiles would get added to non-empty vector!" << std::endl;
            return 1;
        } else {
            int currentProfOffset = 0;
            for (int facetInd = 0; facetInd < facets.size(); ++facetInd) {
                auto &facet = facets[facetInd];
                if (facet.facetProperties.isProfile) {

                    std::vector<Texel> profile;
                    if (!InitializeProfile(facet, profile)) {
                        printf("[ERROR] Initializing facetProf #%d\n", facetInd);
                        exit(0);
                    }

                    // append to a global continuous structure
                    model->profiles.insert(std::end(model->profiles), std::begin(profile), std::end(profile));

                    for (auto &polygonMesh: model->poly_meshes) {
                        for (auto &polygon: polygonMesh->poly) {
                            if (polygon.parentIndex == facetInd) {
                                polygon.profProps.profileOffset = currentProfOffset;
                                polygon.profProps.profileType = facet.facetProperties.profileType;
                            }
                        }
                    }
                    for (auto &triangleMesh: model->triangle_meshes) {
                        for (auto &triangle: triangleMesh->poly) {
                            if (triangle.parentIndex == facetInd) {
                                triangle.profProps.profileOffset = currentProfOffset;
                                triangle.profProps.profileType = facet.facetProperties.profileType;
                            }
                        }
                    }
                    currentProfOffset += PROFILE_SIZE;
                } // one more profile
            }
#ifdef BOUND_CHECK
            model->nbProfSlices_total = currentProfOffset;
#endif
        }

        /*for(auto& mesh : model->triangle_meshes){
            uint32_t triInd = 0;
            for(auto& triangle : mesh->poly){
                *//*if(triangle.facProps.stickingFactor > 0){
                    std::cout << triangle.parentIndex << " | " << triInd << " => ";
                    std::cout << mesh->vertices3d[mesh->indices[triInd].x].x << " , " << mesh->vertices3d[mesh->indices[triInd].x].y << " , " << mesh->vertices3d[mesh->indices[triInd].x].z << " | ";
                    std::cout << mesh->vertices3d[mesh->indices[triInd].y].x << " , " << mesh->vertices3d[mesh->indices[triInd].y].y << " , " << mesh->vertices3d[mesh->indices[triInd].y].z << " | ";
                    std::cout << mesh->vertices3d[mesh->indices[triInd].z].x << " , " << mesh->vertices3d[mesh->indices[triInd].z].y << " , " << mesh->vertices3d[mesh->indices[triInd].z].z << " | ";
                    std::cout << std::endl;
                }*//*
                if(triangle.texProps.textureFlags!= noTexture){
                    std::cout << triangle.parentIndex << " | " << triInd << " => ";
                    std::cout << mesh->vertices3d[mesh->indices[triInd].x].x << " , " << mesh->vertices3d[mesh->indices[triInd].x].y << " , " << mesh->vertices3d[mesh->indices[triInd].x].z << " | ";
                    std::cout << mesh->vertices3d[mesh->indices[triInd].y].x << " , " << mesh->vertices3d[mesh->indices[triInd].y].y << " , " << mesh->vertices3d[mesh->indices[triInd].y].z << " | ";
                    std::cout << mesh->vertices3d[mesh->indices[triInd].z].x << " , " << mesh->vertices3d[mesh->indices[triInd].z].y << " , " << mesh->vertices3d[mesh->indices[triInd].z].z << " | ";
                    std::cout << std::endl;
                }
                triInd++;
            }
        }*/

        return 0;
    }

    //! Load simulation data (geometry etc.) from Molflow's serialization output
    flowgpu::Model *loadFromSerialization(const std::string &inputString) {
        std::stringstream inputStream;
        inputStream << inputString;
        cereal::BinaryInputArchive inputArchive(inputStream);

        flowgpu::Model *model = new flowgpu::Model();
        std::vector<Vector3d> vertices3d;
        std::vector<TempFacet> facets;

        //Worker params
        inputArchive(cereal::make_nvp("wp", model->wp));
        inputArchive(cereal::make_nvp("ontheflyParams", model->ontheflyParams));
        inputArchive(cereal::make_nvp("IDs", model->IDs));
        inputArchive(cereal::make_nvp("parameters", model->parameters));
        //inputArchive(cereal::make_nvp("temperatures",model->temperatures));
        inputArchive(cereal::make_nvp("moments", model->moments));
        //inputArchive(cereal::make_nvp("desorptionParameterIDs",model->desorptionParameterIDs));

        //Geometry
        inputArchive(cereal::make_nvp("GeomProperties", model->geomProperties));
        inputArchive(vertices3d);

        model->structures.resize(model->geomProperties.nbSuper); //Create structures

        //Facets
        facets.resize(model->geomProperties.nbFacet);

        for (int facInd = 0; facInd < model->geomProperties.nbFacet; ++facInd) {
            inputArchive(cereal::make_nvp("facet" + std::to_string(facInd), facets[facInd]));
        }

        std::vector<float3> vertices3f;
        for (std::vector<Vector3d>::iterator it = vertices3d.begin(); it != vertices3d.end(); it++)
            vertices3f.emplace_back(make_float3(it->x, it->y, it->z));
        vertices3d.clear();

        Log::console_msg(3, "#ModelReader: Parsing geometry\n");
        if (parseGeomFromSerialization(model, facets, vertices3f)) {
            delete model;
            model = nullptr;
            return model;
        }
        Log::console_msg(3, "#ModelReader: Gas mass: {}\n", model->wp.gasMass);
        Log::console_msg(3, "#ModelReader: Maxwell: {}\n", model->wp.useMaxwellDistribution);
        Log::console_msg(3, "#ModelReader: Name: {}\n", model->geomProperties.name);
        Log::console_msg(3, "#ModelReader: #Vertex: {}\n", vertices3d.size());
        Log::console_msg(3, "#ModelReader: #Facets: {}\n", model->geomProperties.nbFacet);
        size_t nbTexelCount = 0;
        for (int facInd = 0; facInd < facets.size(); ++facInd) {
            if ((facets[facInd].texelInc.empty() && facets[facInd].facetProperties.isTextured) ||
                (!facets[facInd].texelInc.empty() && !facets[facInd].facetProperties.isTextured)) {
                std::cerr << "#ModelReader: [ERROR] Texture flag and texel increments don't align! " << std::endl;
                exit(0);
            }
            if (facets[facInd].facetProperties.isTextured) {
                Log::console_msg(5, "#ModelReader: Facet# {}  #Texels: {}\n", facInd, facets[facInd].texelInc.size());
                nbTexelCount += facets[facInd].texelInc.size();
            }
        }
        for (auto &mesh: model->triangle_meshes) {
            std::cout << "#ModelReader: #Tri " << mesh->poly.size() << std::endl;
            size_t nbProf = 0;
            size_t nbTex = 0;
            size_t triCount = 0;
            size_t count_2sided = 0;
            size_t count_transparent = 0;
            size_t count_transparent_semi = 0;
            size_t count_neigh = 0;

            for (auto &tri: mesh->poly) {
                if (tri.facProps.endangered_neighbor) ++count_neigh;
                if (tri.facProps.is2sided) ++count_2sided;
                if (tri.facProps.opacity == 0.0) ++count_transparent;
                else if (tri.facProps.opacity > 0.0 && tri.facProps.opacity < 1.0) ++count_transparent_semi;
                if (tri.profProps.profileType != PROFILE_FLAGS::noProfile)
                    ++nbProf;
                if (tri.texProps.textureFlags != TEXTURE_FLAGS::noTexture) {
                    ++nbTex;
                    std::cout << "#ModelReader: Tri#" << triCount++ << "[" << tri.parentIndex << "] #TexOffset: "
                              << tri.texProps.textureOffset << std::endl;
                }
            }
            Log::console_msg(5, "#ModelReader: #SharpNeighborTri {}\n", count_neigh);
            Log::console_msg(5, "#ModelReader: #TransparentTri {}\n", count_transparent);
            Log::console_msg(5, "#ModelReader: #SemiTransparentTri {}\n", count_transparent_semi);
            Log::console_msg(5, "#ModelReader: #ProfiledTris {}\n", nbProf);
            Log::console_msg(5, "#ModelReader: #TexturedTris {}\n", nbTex);
        }

        std::cout << "#ModelReader: #TextureCells: " << model->textures.size() << std::endl;

        if (nbTexelCount != model->textures.size()) {
            std::cerr << "#ModelReader: [ERROR] Texture count out of sync: " << nbTexelCount << " / "
                      << model->textures.size() << std::endl;
            exit(0);
        }
        return model;
    }

    //! Load simulation data (geometry etc.) from Molflow's serialization file output
    int
    loadFromSimModel(std::shared_ptr<Model> &model, std::shared_ptr<flowgpu::MolflowGPUSettings> &settings,
                     const SimulationModel &simModel) {

        Chronometer timer;
        timer.Start();

        model = std::make_shared<flowgpu::Model>();

        if (!settings)
            settings = std::make_shared<flowgpu::MolflowGPUSettings>();

        std::vector<float3> vertices3d;
        std::vector<TempFacet> facets;

        //Worker params
        //inputarchive(cereal::make_nvp("wp",model->parametersGlobal));

        settings->gasMass = simModel.wp.gasMass;
        settings->useMaxwellDistribution = simModel.wp.useMaxwellDistribution;
        model->ontheflyParams = simModel.otfParams;

        model->geomProperties = simModel.sh;
        vertices3d.reserve(simModel.vertices3.size());
        for (auto &vert: simModel.vertices3) {
            vertices3d.emplace_back(make_float3(vert.x, vert.y, vert.z));
        }

        size_t countInd = 0;
        facets.resize(simModel.facets.size());
        int facInd = 0;
        for (auto &fac: simModel.facets) {
            // Create facet and edit by tmp reference
            auto mf_fac = std::dynamic_pointer_cast<MolflowSimFacet>(fac);
            auto &tmp = facets[facInd++];
            tmp.facetProperties = mf_fac->sh;
            tmp.texelInc = mf_fac->textureCellIncrements;
            tmp.indices = mf_fac->indices;
            countInd += tmp.indices.size();
            tmp.vertices2 = mf_fac->vertices2;
            tmp.angleMapPDF = mf_fac->angleMap.pdf;
            tmp.outgassingMap = mf_fac->ogMap.outgassingMap;
        }

        Log::console_msg(3, "#ModelReader: Gas mass: {}\n", settings->gasMass);
        Log::console_msg(3, "#ModelReader: Maxwell: {}\n", settings->useMaxwellDistribution);
        Log::console_msg(3, "#ModelReader: Name: {}\n", model->geomProperties.name);
        Log::console_msg(3, "#ModelReader: #Vertex: {}\n", vertices3d.size());
        Log::console_msg(3, "#ModelReader: #Indices: {}\n", countInd);
        Log::console_msg(3, "#ModelReader: #Facets: {}\n", model->geomProperties.nbFacet);

        for (int facInd = 0; facInd < facets.size(); ++facInd) {
            if (!facets[facInd].texelInc.empty())
                Log::console_msg(5, "#ModelReader: Facet# {}  #Texels: {}\n", facInd, facets[facInd].texelInc.size());
        }

        timer.ReInit(); timer.Start();
        Log::console_header(3, "#ModelReader: Parsing geometry\n");
        parseGeomFromSerialization(model.get(), facets, vertices3d);
        Log::console_footer(3, "#ModelReader: Done parsing geometry {}s\n", timer.Elapsed());

        std::vector<CommonEdge> edges;
        std::vector<Facet *> facet_ptr;
        facet_ptr.resize(simModel.facets.size());
        std::transform(simModel.facets.begin(), simModel.facets.end(), facet_ptr.begin(),
                       [](const std::shared_ptr<SimulationFacet> &f) { return (Facet *) (f.get()); }
        );


        timer.ReInit(); timer.Start();
        Log::console_msg(3, "#ModelReader: Getting overlapping edges\n");
        std::vector<OverlappingEdge> edges_overlap;
        /*NeighborScan::GetOverlappingEdges(facet_ptr, simModel.vertices3, edges_overlap);
        int nOverlapEdges = edges_overlap.size();
        std::vector<CommonEdge> edges_un;
        Log::console_msg(3, "#ModelReader: Got overlapping edges {}s\n", timer.Elapsed());
        timer.ReInit(); timer.Start();
        NeighborScan::GetAnalysedUnorientedCommonEdges(facet_ptr, edges_un);
        //NeighborScan::GetAnalysedCommonEdges(facet_ptr, edges);
        // compare edges
        Log::console_msg(3, "#ModelReader: Got unoriented edges {}s\n", timer.Elapsed());

        timer.ReInit(); timer.Start();
        NeighborScan::CompareOverlap(edges_overlap, edges_un);
        // --- end ---
        Log::console_msg(3, "#ModelReader: Compared overlapping edges {}s\n", timer.Elapsed());
*/
        /*if(NeighborScan::GetAnalysedUnorientedCommonEdges(facet_ptr, edges)) {
            for (auto &edge : edges_overlap) {
                auto id1 = edge.facetId1;
                auto id2 = edge.facetId2;
                auto angle = edge.angle;*/

        timer.ReInit(); timer.Start();
        //edges.clear();
        edges_overlap.clear();

        if (NeighborScan::GetAnalysedOverlappingEdges(facet_ptr, simModel.vertices3, edges_overlap)) {
            for (auto &edge: edges_overlap) {
                auto id1 = edge.facetId1;
                auto id2 = edge.facetId2;
                auto angle = edge.angle;

                if (angle > DegToRad(89.0) && angle < DegToRad(180.0)) { // sharp angle
                    // label corresponding facets
                    for (auto &poly: model->triangle_meshes.front()->poly) {
                        if (poly.parentIndex == id1 || poly.parentIndex == id2) {
                            poly.facProps.endangered_neighbor = true;
                            poly.facProps.offset_factor
                                    = 1.0;//std::max(poly.facProps.offset_factor,(float)(((angle - (DegToRad(89.0))) / (M_PI - DegToRad(89.0))))); // normalize offset to 90 deg radiant value and reverse factor 0->1,1->0
                            poly.facProps.min_angle = std::min(poly.facProps.min_angle, (float) (RadToDeg(angle)));
                            poly.facProps.max_angle = std::max(poly.facProps.max_angle, (float) (RadToDeg(angle)));
                        }
                    }
                }
                Log::console_msg(5, " Poly #{} -- #{} with angle {}\n", id1, id2, (float) (RadToDeg(angle)));
            }
        }

        Log::console_msg(3, "#ModelReader: Analysed edges {}s\n", timer.Elapsed());

        /*if(NeighborScan::GetAnalysedCommonEdges(facet_ptr, edges)) {
            for (auto &edge : edges) {
                auto id1 = edge.facetId[0];
                auto id2 = edge.facetId[1];
                auto angle = edge.angle;

                if(1){//if(angle > 90.0 / 180.0*M_PI || angle < 180.0 / 180.0*M_PI) { // sharp angle
                    // label corresponding facets
                    for(auto& poly : model->triangle_meshes.front()->poly) {
                        if(poly.parentIndex == id1 || poly.parentIndex == id2) {
                            poly.facProps.endangered_neighbor = true;
                            poly.facProps.offset_factor = std::max(poly.facProps.offset_factor,(float)(1.0 - ((angle - (0.5 * M_PI)) / (M_PI - 0.5 * M_PI)))); // normalize offset to 90 deg radiant value and reverse factor 0->1,1->0
                            poly.facProps.min_angle = std::min(poly.facProps.min_angle, (float)(angle * 180 / M_PI));
                            poly.facProps.max_angle = std::max(poly.facProps.max_angle, (float)(angle * 180 / M_PI));
                        }
                    }
                }
            }
        }*/

        Log::console_msg(3, "#ModelReader: Listing endangered neighbors\n");
        int facetIndex = 0;
        for (auto &poly: model->triangle_meshes.front()->poly) {
            if (poly.facProps.endangered_neighbor)
                Log::console_msg(5, "Facet {} ({}) with offset {} [{} , {}]\n",
                                 facetIndex, poly.parentIndex, poly.facProps.offset_factor, poly.facProps.min_angle,
                                 poly.facProps.max_angle);
            ++facetIndex;
        }

        for (auto &mesh: model->triangle_meshes) {
            std::cout << "#ModelReader: #Tri " << mesh->poly.size() << std::endl;
            size_t nbProf = 0;
            size_t nbTex = 0;
            size_t triCount = 0;
            size_t count_2sided = 0;
            size_t count_transparent = 0;
            size_t count_transparent_semi = 0;
            size_t count_neigh = 0;

            for (auto &tri: mesh->poly) {
                if (tri.facProps.endangered_neighbor) ++count_neigh;
                if (tri.facProps.is2sided) ++count_2sided;
                if (tri.facProps.opacity == 0.0) ++count_transparent;
                else if (tri.facProps.opacity > 0.0 && tri.facProps.opacity < 1.0) ++count_transparent_semi;
                if (tri.profProps.profileType != PROFILE_FLAGS::noProfile)
                    ++nbProf;
                if (tri.texProps.textureFlags != TEXTURE_FLAGS::noTexture) {
                    ++nbTex;
                    Log::console_msg(5, "#ModelReader: Tri# {} [{}] #TexOffset: {}\n", triCount++, tri.parentIndex,
                                     tri.texProps.textureOffset);
                }
            }
            Log::console_msg(5, "#ModelReader: #SharpNeighborTri {}/{}\n", count_neigh, mesh->poly.size());
            Log::console_msg(5, "#ModelReader: #TransparentTri {}\n", count_transparent);
            Log::console_msg(5, "#ModelReader: #SemiTransparentTri {}\n", count_transparent_semi);
            Log::console_msg(5, "#ModelReader: #ProfiledTris {}\n", nbProf);
            Log::console_msg(5, "#ModelReader: #TexturedTris {}\n", nbTex);
        }

        Log::console_msg(5, "#ModelReader: #TextureCells {}\n", model->textures.size());
        /*if (nbTexelCount != model->textures.size()) {
            std::cerr << "#ModelReader: [ERROR] Texture count out of sync: " << nbTexelCount << " / "
                      << model->textures.size() << std::endl;
            exit(0);
        }*/

        constexpr size_t cdf_size = 100; // points in a cumulative distribution function
        for (auto &cdfs: ((MolflowSimulationModel *) (&simModel))->tdParams.CDFs) {
            for (auto &cdf: cdfs) {
                model->cdfs_1.emplace_back(cdf.first);
                model->cdfs_2.emplace_back(cdf.second);
            }
        }
        return 0;
    }
}// namespace