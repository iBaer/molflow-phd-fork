//
// Created by Pascal Baehr on 28.04.20.
//

#ifndef MOLFLOW_PROJ_GEOMETRYSIMU_H
#define MOLFLOW_PROJ_GEOMETRYSIMU_H

#include <vector>
#include "MolflowTypes.h"
#include "Buffer_shared.h"
#include "Parameter.h"

#ifdef WIN32
#include <mutex>
#endif

struct SubprocessFacet;
class SuperStructure;

struct TimeDependentParamters {
    std::vector<Distribution2D> parameters;

    std::vector<std::vector<std::pair<double, double>>> CDFs; //cumulative distribution function for each temperature
    std::vector<std::vector<std::pair<double, double>>> IDs; //integrated distribution function for each time-dependent desorption type
    std::vector<Moment> moments;             //moments when a time-dependent simulation state is recorded
    /*std::vector<UserMoment> userMoments;    //user-defined text values for defining time moments (can be time or time series)
    std::vector<double> temperatures; //keeping track of all temperatures that have a CDF already generated
    std::vector<size_t> desorptionParameterIDs; //time-dependent parameters which are used as desorptions, therefore need to be integrated
*/
};

struct SimulationModel {
public:
    SimulationModel() : otfParams(), tdParams(), wp(), sh(){};
    void CalculateFacetParams(SubprocessFacet* f);

    // Sim functions
    double GetOpacityAt(SubprocessFacet *f, double time) const;

    // Geometry Description
    std::vector<std::vector<SubprocessFacet>>    facets;    // All facets of this geometry
    std::vector<SuperStructure> structures;
    std::vector<Vector3d> vertices3; // Vertices (3D space)

    // Simulation Properties
    OntheflySimulationParams otfParams;
    TimeDependentParamters tdParams;
    WorkerParams wp;

    // Geometry Properties
    GeomProperties sh;
};

class Anglemap {
public:
    std::vector<size_t>   pdf;		  // Incident angle distribution, phi and theta, not normalized. Used either for recording or for 2nd order interpolation
    std::vector<double>   phi_CDFs;    // A table containing phi distributions for each theta, starting from 0 for every line (1 line = 1 theta value). For speed we keep it in one memory block, 1 pointer
    std::vector<size_t>   phi_CDFsums; // since CDF runs only to the middle of the last segment, for each theta a line sum is stored here. Also a pdf for theta
    std::vector<double>   theta_CDF;	  // Theta CDF, not normalized. nth value is the CDF at the end of region n (beginning of first section is always 0)
    size_t   theta_CDFsum; // since theta CDF only runs till the middle of the last segment, the map sum is here
};

// Local facet structure
struct SubprocessFacet{
    SubprocessFacet();
    SubprocessFacet(size_t nbIndex);

    FacetProperties sh;

    std::vector<size_t>      indices;          // Indices (Reference to geometry vertex)
    std::vector<Vector2d> vertices2;        // Vertices (2D plane space, UV coordinates)
    std::vector<double>   textureCellIncrements;              // Texure increment
    std::vector<bool>     largeEnough;      // cells that are NOT too small for autoscaling
    double   fullSizeInc;       // Texture increment of a full texture element
    std::vector<double>   outgassingMap; // Cumulative outgassing map when desorption is based on imported file
    double outgassingMapWidthD; //actual outgassing file map width
    double outgassingMapHeightD; //actual outgassing file map height

    Anglemap angleMap; //TODO: -> GeneratingAngleMap from 2.7

    // Used for texture init only
    double rw;
    double rh;
    double iw;
    double ih;

    // Temporary var (used in FillHit for hit recording)
    bool   isReady;         // Volatile state
    size_t    textureSize;   // Texture size (in bytes)
    size_t    profileSize;   // profile size (in bytes)
    size_t    directionSize; // direction field size (in bytes)
    size_t    angleMapSize;  // incidentangle map size (in bytes)

    size_t globalId; //Global index (to identify when superstructures are present)

    // Facet hit counters
    //std::vector<FacetHitBuffer> tmpCounter; //1+nbMoment
    //std::vector<FacetHistogramBuffer> tmpHistograms; //1+nbMoment

    //void ResetCounter();
    //void ResizeCounter(size_t nbMoments);
    bool InitializeOnLoad(const size_t &id, const size_t &nbMoments, size_t &histogramTotalSize);

    void InitializeHistogram(const size_t &nbMoments, size_t &histogramTotalSize);

    bool InitializeDirectionTexture(const size_t &nbMoments);

    bool InitializeProfile(const size_t &nbMoments);

    bool InitializeTexture(const size_t &nbMoments);

    bool InitializeAngleMap();

    void InitializeOutgassingMap();

    bool InitializeLinkAndVolatile(const size_t & id);

    size_t GetHitsSize(size_t nbMoments) const;
    //void RegisterTransparentPass(SubprocessFacet *facet); //Allows one shared Intersect routine between MolFlow and Synrad

    template<class Archive> void serialize(Archive& archive) {
        archive(
            sh,
            indices,
            vertices2,
            outgassingMap,
            angleMap.pdf,
            textureCellIncrements
        );
    }

};

// Local simulation structure

class AABBNODE;

class SuperStructure {
public:
    SuperStructure();
    ~SuperStructure();
    std::vector<SubprocessFacet>  facets;   // Facet handles
    AABBNODE* aabbTree; // Structure AABB tree
};

/*!
 * @brief One instance is the state for one facet for a single moment
 */
class FacetMomentSnapshot {
public:
    FacetMomentSnapshot& operator+=(const FacetMomentSnapshot& rhs);
    FacetMomentSnapshot& operator+(const FacetMomentSnapshot& rhs);
    FacetHitBuffer hits;
    std::vector<ProfileSlice> profile;
    std::vector<TextureCell> texture;
    std::vector<DirectionCell> direction;
    FacetHistogramBuffer histogram;
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
                hits,
                profile,
                texture,
                direction,
                histogram
        );
    }
};

class FacetState {
public:
    FacetState& operator+=(const FacetState& rhs);
    std::vector<size_t> recordedAngleMapPdf; //Not time-dependent
    std::vector<FacetMomentSnapshot> momentResults; //1+nbMoment
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(
                recordedAngleMapPdf,
                momentResults
        );
    }
};

class GlobalSimuState { //replaces old hits dataport
public:
    GlobalSimuState& operator=(const GlobalSimuState& src);
    GlobalSimuState& operator+=(const GlobalSimuState& src);
    bool initialized = false;
    void clear();
    void Resize(const SimulationModel &model);
    void Reset();

#if defined(MOLFLOW)
    GlobalHitBuffer globalHits;
    std::vector<FacetHistogramBuffer> globalHistograms; //1+nbMoment
    std::vector<FacetState> facetStates; //nbFacet
#endif
    //std::timed_mutex mutex;
};
#endif //MOLFLOW_PROJ_GEOMETRYSIMU_H
