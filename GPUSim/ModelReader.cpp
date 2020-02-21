//
// Created by pbahr on 12/02/2020.
//

#include "ModelReader.h"
#include "Poly2TriConverter.h"
//#include "Facet_shared.h"

// debug output
#include <fstream>

#include "helper_math.h"
#include "../MolflowTypes.h"

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

template<class Archive>
void serialize(Archive & archive,
               float2 & m)
{
    archive( m.x, m.y);
}

template<class Archive>
void serialize(Archive & archive,
               float3 & m)
{
    archive( m.x, m.y, m.z);
}

template<class Archive>
void serialize(Archive & archive,
               int3 & m)
{
    archive( m.x, m.y, m.z);
}

/*! \namespace flowgeom - Molflow Geometry code */
namespace flowgeom {

/*! --- Initialise model with a Molflow-exported geometry --- */
    flowgpu::Model* initializeModel(std::string fileName){

    std::cout << "#GPUTestsuite: Loading input file: " << fileName << std::endl;

    flowgpu::Model* model = new flowgpu::Model();
    std::ifstream file( fileName );
    cereal::XMLInputArchive archive( file );

#ifdef WITHTRIANGLES
    model->triangle_meshes.push_back(new flowgpu::TriangleMesh());
    archive(
            cereal::make_nvp("poly", model->triangle_meshes[0]->poly) ,
            cereal::make_nvp("facetProbabilities", model->triangle_meshes[0]->facetProbabilities) ,
            cereal::make_nvp("cdfs", model->triangle_meshes[0]->cdfs) ,
            //cereal::make_nvp("vertices2d", nullptr) ,
            cereal::make_nvp("vertices3d", model->triangle_meshes[0]->vertices3d) ,
            cereal::make_nvp("indices", model->triangle_meshes[0]->indices) ,
            cereal::make_nvp("nbFacets", model->triangle_meshes[0]->nbFacets) ,
            cereal::make_nvp("nbVertices", model->triangle_meshes[0]->nbVertices) ,
            cereal::make_nvp("nbFacetsTotal", model->nbFacets_total) ,
            cereal::make_nvp("nbVerticesTotal", model->nbVertices_total) ,
            cereal::make_nvp("useMaxwellDistribution", model->parametersGlobal.useMaxwellDistribution)
    );
#else
        model->poly_meshes.push_back(new flowgpu::PolygonMesh());
        archive(
                cereal::make_nvp("poly", model->poly_meshes[0]->poly) ,
                cereal::make_nvp("facetProbabilities", model->poly_meshes[0]->facetProbabilities) ,
                cereal::make_nvp("cdfs", model->poly_meshes[0]->cdfs) ,
                cereal::make_nvp("vertices2d", model->poly_meshes[0]->vertices2d) ,
                cereal::make_nvp("vertices3d", model->poly_meshes[0]->vertices3d) ,
                cereal::make_nvp("indices", model->poly_meshes[0]->indices) ,
                cereal::make_nvp("nbFacets", model->poly_meshes[0]->nbFacets) ,
                cereal::make_nvp("nbVertices", model->poly_meshes[0]->nbVertices) ,
                cereal::make_nvp("nbFacetsTotal", model->nbFacets_total) ,
                cereal::make_nvp("nbVerticesTotal", model->nbVertices_total) ,
                cereal::make_nvp("useMaxwellDistribution", model->parametersGlobal.useMaxwellDistribution)
        );
#endif

        std::cout << "#GPUTestsuite: Loading completed!" << std::endl;

        return model;
    }

    void convertFacet2Poly(const std::vector<TempFacet>& facets, std::vector<flowgeom::Polygon>& convertedPolygons){

        int32_t vertCount = 0;
        for(int i = 0; i < facets.size(); ++i){
            auto& temp = facets[i];

            flowgeom::Polygon polygon(temp.vertices2.size());
            polygon.stickingFactor = temp.facetProperties.sticking;

            if(polygon.nbVertices != temp.facetProperties.nbIndex){
                polygon.nbVertices = temp.facetProperties.nbIndex;
                std::cout << "Parsing error! Vert size != nbIndex"<<std::endl;
                exit(0);
            }

            polygon.indexOffset = vertCount;
            polygon.O = temp.facetProperties.O;
            polygon.U = temp.facetProperties.U;
            polygon.V = temp.facetProperties.V;
            polygon.Nuv = temp.facetProperties.Nuv;
            polygon.nU = temp.facetProperties.nU;
            polygon.nV = temp.facetProperties.nV;
            polygon.N = temp.facetProperties.N;

            polygon.parentIndex = i;

            convertedPolygons.push_back(std::move(polygon));

            vertCount += polygon.nbVertices;
        }

    }

    //! Calculate outgassing values in relation to (tri_area / poly_area)
    void CalculateRelativeTriangleOutgassing(const std::vector<TempFacet>& facets, flowgpu::TriangleMesh* triMesh){
        float fullOutgassing = 0;
        int facetIndex = 0;
        for(auto& facet : triMesh->poly){

            // Calculate triangle area
            auto& triIndices = triMesh->indices[facetIndex];
            auto& a = triMesh->vertices3d[triIndices.x];
            auto& b = triMesh->vertices3d[triIndices.y];
            auto& c = triMesh->vertices3d[triIndices.z];


            float3 ab = make_float3((b.x - a.x) , (b.y - a.y) , (b.z - a.z));
            float3 ac = make_float3((c.x - a.x) , (c.y - a.y) , (c.z - a.z));
            float area = 0.5 * length(cross(ab,ac));

            // outgassing of a triangle is only a percentage of the original polygon's
            float areaPercentageOfPoly = (area/facets[facet.parentIndex].facetProperties.area);

            float fullOutgassing_inc = fullOutgassing + (facets[facet.parentIndex].facetProperties.outgassing * areaPercentageOfPoly)/ (1.38E-23*facets[facet.parentIndex].facetProperties.temperature);
            triMesh->facetProbabilities.push_back(make_float2(fullOutgassing, fullOutgassing_inc));
            fullOutgassing = fullOutgassing_inc;

            ++facetIndex;
        }
        for(auto& facetProb : triMesh->facetProbabilities){
            facetProb.x /= fullOutgassing; // normalize to [0,1]
            facetProb.y /= fullOutgassing; // normalize to [0,1]
        }
    };

    std::vector<TextureCell> InitializeTexture(TempFacet& facet)
    {
        std::vector<TextureCell> texture;
        std::vector<double> textureCellIncrements;

        //Textures
        if (facet.facetProperties.isTextured) {
            size_t nbE = facet.facetProperties.texWidth*facet.facetProperties.texHeight;
            try {
                texture = std::vector<TextureCell>(nbE);
            }
            catch (...) {
                printf("Not enough memory to load textures\n");
                //return nullptr;
            }
        }
        return texture;
    }

    //! Load simulation data (geometry etc.) from Molflow's serialization output
    flowgpu::Model* loadFromSerialization(std::string fileName){
        {
            std::ifstream file( fileName );
            cereal::XMLInputArchive inputarchive(file);

            flowgpu::Model* model = new flowgpu::Model();
            std::vector<float3> vertices3d;
            std::vector<TempFacet> facets;

            //Worker params
            //inputarchive(cereal::make_nvp("wp",model->parametersGlobal));
            inputarchive(cereal::make_nvp("gasMass",model->parametersGlobal.gasMass));
            inputarchive(cereal::make_nvp("useMaxwellDistribution",model->parametersGlobal.useMaxwellDistribution));
            inputarchive(cereal::make_nvp("GeomProperties",model->geomProperties));
            inputarchive(vertices3d);
            facets.resize(model->geomProperties.nbFacet);
            for(int facInd = 0; facInd < model->geomProperties.nbFacet;++facInd){
                inputarchive(cereal::make_nvp("facet"+std::to_string(facInd),facets[facInd]));
            }



            std::cout << "#ModelReader: Gas mass: " << model->parametersGlobal.gasMass << std::endl;
            std::cout << "#ModelReader: Maxwell: " << model->parametersGlobal.useMaxwellDistribution << std::endl;
            std::cout << "#ModelReader: Name: " << model->geomProperties.name << std::endl;
            std::cout << "#ModelReader: #Vertex: " << vertices3d.size() << std::endl;
            std::cout << "#ModelReader: #Facets: " << model->geomProperties.nbFacet << std::endl;

            // First create a regular polygonmesh
            // transform Molflow facet data to simulation polygons
            // transform polygons to triangles
            flowgpu::PolygonMesh* polyMesh = new flowgpu::PolygonMesh();
            convertFacet2Poly(facets,polyMesh->poly);

            int i= 0;
            int indexOffset = 0;

            for(auto& facet : facets){
                polyMesh->poly[i++].indexOffset = indexOffset;
                indexOffset += facet.indices.size();

                for(auto ind : facet.indices){
                    polyMesh->indices.push_back(ind);
                }
                for(auto vert : facet.vertices2){
                    polyMesh->vertices2d.push_back(vert);
                }
            }
            polyMesh->vertices3d = vertices3d;

            polyMesh->nbFacets = model->geomProperties.nbFacet;
            polyMesh->nbVertices = model->geomProperties.nbVertex;

            // Now create Triangle Mesh
            flowgpu::TriangleMesh* triMesh = new flowgpu::TriangleMesh();
            int nbTris = Poly2TriConverter::PolygonsToTriangles(polyMesh, triMesh);
            triMesh->vertices3d = polyMesh->vertices3d;
            triMesh->nbVertices = triMesh->poly.size() * 3;
            triMesh->nbFacets = triMesh->poly.size();

            triMesh->cdfs.push_back(0);


            if(!polyMesh->poly.empty())
                model->poly_meshes.push_back(polyMesh);
            model->triangle_meshes.push_back(triMesh);

            for(auto& polyMesh : model->poly_meshes){
                model->nbFacets_total += polyMesh->nbFacets;
                model->nbVertices_total += polyMesh->nbVertices;
            }
            for(auto& triMesh : model->triangle_meshes){
                model->nbFacets_total += triMesh->nbFacets;
                model->nbVertices_total += triMesh->nbVertices;
            }
            model->geomProperties.nbFacet = model->nbFacets_total;
            model->geomProperties.nbVertex = model->nbVertices_total;

            //--- Calculate outgassing values in relation to (tri_area / poly_area)
            CalculateRelativeTriangleOutgassing(facets,triMesh);


            if(!model->textures.empty()){
                std::cout << "[WARNING] Textures get added to non-empty vector!"<< std::endl;
                return nullptr;
            }
            else{
                int textureOffset = 0;
                for(int facetInd = 0; facetInd < facets.size(); ++facetInd){
                    auto& facet = facets[facetInd];
                    if(facet.facetProperties.isTextured){
                        std::vector<TextureCell> texture = InitializeTexture(facet);
                        model->textures.insert(std::end(model->textures),std::begin(texture),std::end(texture));
                        for(auto& polyMesh : model->poly_meshes){
                            for(auto& polygon : polyMesh->poly){
                                if(polygon.parentIndex == facetInd){
                                    polygon.textureOffset = textureOffset;
                                    polygon.textureSize = texture.size();
                                    if(facet.facetProperties.countAbs)
                                        polygon.textureFlags |= TextureCounters::countAbs;
                                    if(facet.facetProperties.countRefl)
                                        polygon.textureFlags |= TextureCounters::countRefl;
                                    if(facet.facetProperties.countTrans)
                                        polygon.textureFlags |= TextureCounters::countTrans;
                                    if(facet.facetProperties.countDirection)
                                        polygon.textureFlags |= TextureCounters::countDirection;
                                    if(facet.facetProperties.countDes)
                                        polygon.textureFlags |= TextureCounters::countDes;
                                }
                            }
                        }
                        for(auto& triMesh : model->triangle_meshes){
                            for(auto& triangle : triMesh->poly){
                                if(triangle.parentIndex == facetInd){
                                    triangle.textureOffset = textureOffset;
                                    triangle.textureSize = texture.size();
                                    if(facet.facetProperties.countAbs)
                                        triangle.textureFlags |= TextureCounters::countAbs;
                                    if(facet.facetProperties.countRefl)
                                        triangle.textureFlags |= TextureCounters::countRefl;
                                    if(facet.facetProperties.countTrans)
                                        triangle.textureFlags |= TextureCounters::countTrans;
                                    if(facet.facetProperties.countDirection)
                                        triangle.textureFlags |= TextureCounters::countDirection;
                                    if(facet.facetProperties.countDes)
                                        triangle.textureFlags |= TextureCounters::countDes;
                                }
                            }
                        }
                        textureOffset += texture.size();
                    }
                }
            }
            std::cout << "#ModelReader: #TextureCells: " << model->textures.size() << std::endl;

            return model;
            /*inputarchive(sHandle->ontheflyParams);
            inputarchive(sHandle->CDFs);
            inputarchive(sHandle->IDs);
            inputarchive(sHandle->parameters);
            inputarchive(sHandle->temperatures);
            inputarchive(sHandle->moments);
            inputarchive(sHandle->desorptionParameterIDs);

            //Geometry


            sHandle->structures.resize(sHandle->sh.nbSuper); //Create structures

            //Facets
            for (size_t i = 0; i < sHandle->sh.nbFacet; i++) { //Necessary because facets is not (yet) a vector in the interface
                SubprocessFacet f;
                inputarchive(
                        f.sh,
                        f.indices,
                        f.vertices2,
                        f.outgassingMap,
                        f.angleMap.pdf,
                        f.textureCellIncrements
                );

                //Some initialization
                if (!f.InitializeOnLoad(i)) return false;
                if (f.sh.superIdx == -1) { //Facet in all structures
                    for (auto& s : sHandle->structures) {
                        s.facets.push_back(f);
                    }
                }
                else {
                    sHandle->structures[f.sh.superIdx].facets.push_back(f); //Assign to structure
                }
            }*/
        }//inputarchive goes out of scope, file released
    }
}