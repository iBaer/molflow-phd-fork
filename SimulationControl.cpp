/*
File:        SimulationControl.c
Description: Simulation control routines
Program:     MolFlow
Author:      R. KERSEVAN / J-L PONS / M ADY
Copyright:   E.S.R.F / CERN

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*/

#ifdef WIN
#include <windows.h> // For GetTickCount()
#include <Process.h> // For _getpid()
#else
#include <time.h>
#include <sys/time.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Simulation.h"
#include "Random.h"

// -------------------------------------------------------
// Global handles
// -------------------------------------------------------

FACET     **THits;
SIMULATION *sHandle;

// -------------------------------------------------------
// Timing stuff
// -------------------------------------------------------

#ifdef WIN
BOOL usePerfCounter;         // Performance counter usage
LARGE_INTEGER perfTickStart; // First tick
double perfTicksPerSec;      // Performance counter (number of tick per second)
#endif
DWORD tickStart;

// -------------------------------------------------------

void InitSimulation() {

	// Global handle allocation
	sHandle = (SIMULATION *)malloc(sizeof(SIMULATION));
	memset(sHandle, 0, sizeof(SIMULATION));
	THits = (FACET **)malloc(MAX_THIT * sizeof(FACET *)); // Transparent hit cache

#ifdef WIN
	{
		LARGE_INTEGER qwTicksPerSec;
		usePerfCounter = QueryPerformanceFrequency(&qwTicksPerSec);
		if (usePerfCounter) {
			QueryPerformanceCounter(&perfTickStart);
			perfTicksPerSec = (double)qwTicksPerSec.QuadPart;
		}
		tickStart = GetTickCount();
	}
#else
	tickStart = (DWORD)time(NULL);
#endif

}

// -------------------------------------------------------

void ClearSimulation() {

	int i, j;

	// Free old stuff
	sHandle->CDFs = std::vector<std::vector<std::pair<double, double>>>(); //clear CDF distributions

	SAFE_FREE(sHandle->vertices3);
	for (j = 0; j < sHandle->nbSuper; j++) {
		for (i = 0; i < sHandle->str[j].nbFacet; i++) {
			FACET *f = sHandle->str[j].facets[i];
			if (f) {
				SAFE_FREE(f->indices);
				SAFE_FREE(f->vertices2);
				//SAFE_FREE(f->fullElem);
				SAFE_FREE(f->inc);
				SAFE_FREE(f->largeEnough);
				for (size_t m = 0; m < (sHandle->nbMoments + 1); m++) {

					if (f->hits) SAFE_FREE(f->hits[m]);
					if (f->profile) SAFE_FREE(f->profile[m]);
					if (f->direction) SAFE_FREE(f->direction[m]);
					//if (f->velocityHistogram) SAFE_FREE(f->velocityHistogram);
				}

				SAFE_FREE(f->hits);
				SAFE_FREE(f->profile);
				SAFE_FREE(f->direction);
				//SAFE_FREE(f->velocityHistogram);

				delete(f); f = NULL;
			}

		}
		SAFE_FREE(sHandle->str[j].facets);
		if (sHandle->str[j].aabbTree) {
			DestroyAABB(sHandle->str[j].aabbTree->left);
			DestroyAABB(sHandle->str[j].aabbTree->right);
			free(sHandle->str[j].aabbTree);
			sHandle->str[j].aabbTree = NULL;
		}
	}

	sHandle->CDFs = std::vector<std::vector<std::pair<double, double>>>();
	sHandle->IDs = std::vector<std::vector<std::pair<double, double>>>();
	sHandle->parameters = std::vector<Parameter>();
	sHandle->temperatures = std::vector<double>();
	sHandle->moments = std::vector<double>();
	sHandle->desorptionParameterIDs = std::vector<size_t>();

	ClearACMatrix();
	memset(sHandle, 0, sizeof(SIMULATION));

}
// -------------------------------------------------------

DWORD RevertBit(DWORD dw) {
	DWORD dwIn = dw;
	DWORD dwOut = 0;
	DWORD dWMask = 1;
	int i;
	for (i = 0; i < 32; i++) {
		if (dwIn & 0x80000000UL) dwOut |= dWMask;
		dwIn = dwIn << 1;
		dWMask = dWMask << 1;
	}
	return dwOut;
}

DWORD GetSeed() {

	/*#ifdef WIN
	DWORD r;
	_asm {
	rdtsc
	mov r,eax
	}
	return RevertBit(r ^ (DWORD)(_getpid()*65519));
	#else*/
	return (DWORD)((int)(GetTick()*1000.0)*_getpid());
	//#endif

}



/*double Norme(Vector3d *v) { //already defined
	return sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
}*/



BOOL LoadSimulation(Dataport *loader) {

	int i, j, idx;
	BYTE *buffer;
	BYTE *incBuff;
	BYTE *bufferStart;
	SHGEOM *shGeom;
	Vector3d *shVert;
	double t1, t0;
	DWORD seed;
	char err[128];

	t0 = GetTick();

	sHandle->loadOK = FALSE;
	SetState(PROCESS_STARTING, "Clearing previous simulation");
	ClearSimulation();
	
	/* //Mutex not necessary: by the time the COMMAND_LOAD is issued the interface releases the handle, concurrent reading is safe and it's only destroyed by the interface when all processes are ready loading
	   //Result: faster, parallel loading
	// Connect the dataport
	SetState(PROCESS_STARTING, "Waiting for 'loader' dataport access...");
	if (!AccessDataportTimed(loader,15000)) {
		SetErrorSub("Failed to connect to DP");
		return FALSE;
	}
	*/

	SetState(PROCESS_STARTING, "Loading simulation");

	buffer = (BYTE *)loader->buff;
	bufferStart = buffer; //memorize start for later


	// Load new geom from the dataport

	shGeom = (SHGEOM *)buffer;
	if (shGeom->nbSuper > MAX_STRUCT) {
		ReleaseDataport(loader);
		SetErrorSub("Too many structures");
		return FALSE;
	}
	if (shGeom->nbSuper <= 0) {
		ReleaseDataport(loader);
		SetErrorSub("No structures");

		return FALSE;
	}



	sHandle->nbVertex = shGeom->nbVertex;
	sHandle->nbSuper = shGeom->nbSuper;
	sHandle->totalFacet = shGeom->nbFacet;
	sHandle->nbMoments = shGeom->nbMoments;
	sHandle->latestMoment = shGeom->latestMoment;
	sHandle->totalDesorbedMolecules = shGeom->totalDesorbedMolecules;
	sHandle->finalOutgassingRate = shGeom->finalOutgassingRate;
	sHandle->gasMass = shGeom->gasMass;
	sHandle->enableDecay = shGeom->enableDecay;
	sHandle->halfLife = shGeom->halfLife;
	sHandle->timeWindowSize = shGeom->timeWindowSize;
	sHandle->useMaxwellDistribution = shGeom->useMaxwellDistribution;
	sHandle->calcConstantFlow = shGeom->calcConstantFlow;
	sHandle->motionType = shGeom->motionType;
	sHandle->motionVector1 = shGeom->motionVector1;
	sHandle->motionVector2 = shGeom->motionVector2;
	// Prepare super structure (allocate memory for facets)
	buffer += sizeof(SHGEOM) + sizeof(Vector3d)*sHandle->nbVertex;
	for (i = 0; i < sHandle->totalFacet; i++) {
		SHFACET *shFacet = (SHFACET *)buffer;
		sHandle->str[shFacet->superIdx].nbFacet++;
		buffer += sizeof(SHFACET) + shFacet->nbIndex*(sizeof(int) + sizeof(Vector2d));
		if (shFacet->useOutgassingFile) buffer += sizeof(double)*shFacet->outgassingMapWidth*shFacet->outgassingMapHeight;
	}
	for (i = 0; i < sHandle->nbSuper; i++) {
		int nbF = sHandle->str[i].nbFacet;
		if (nbF == 0) {
			/*ReleaseDataport(loader);
			sprintf(err,"Structure #%d has no facets!",i+1);
			SetErrorSub(err);
			return FALSE;*/
		}
		else {

			sHandle->str[i].facets = (FACET **)malloc(nbF * sizeof(FACET *));
			memset(sHandle->str[i].facets, 0, nbF * sizeof(FACET *));
			//sHandle->str[i].nbFacet = 0;
		}
		sHandle->str[i].nbFacet = 0;
	}

	incBuff = buffer; //after facets and outgassing maps, with inc values


	//buffer = (BYTE *)loader->buff;
	buffer = bufferStart; //start from beginning again

	// Name
	memcpy(sHandle->name, shGeom->name, 64);

	// Vertices

	sHandle->vertices3 = (Vector3d *)malloc(sHandle->nbVertex * sizeof(Vector3d));
	if (!sHandle->vertices3) {
		SetErrorSub("Not enough memory to load vertices");
		return FALSE;
	}
	buffer += sizeof(SHGEOM);

	shVert = (Vector3d *)(buffer);
	memcpy(sHandle->vertices3, shVert, sHandle->nbVertex * sizeof(Vector3d));
	buffer += sizeof(Vector3d)*sHandle->nbVertex;

	// Facets
	for (i = 0; i < sHandle->totalFacet; i++) {

		SHFACET *shFacet = (SHFACET *)buffer;
		FACET *f = new FACET;
		if (!f) {
			SetErrorSub("Not enough memory to load facets");
			return FALSE;
		}
		memset(f, 0, sizeof(FACET));
		memcpy(&(f->sh), shFacet, sizeof(SHFACET));
		f->ResizeCounter(sHandle->nbMoments); //Initialize counter

		//f->sh.maxSpeed = 4.0 * sqrt(2.0*8.31*f->sh.temperature/0.001/sHandle->gasMass);

		sHandle->hasVolatile |= f->sh.isVolatile;
		sHandle->hasDirection |= f->sh.countDirection;

		idx = f->sh.superIdx;
		sHandle->str[idx].facets[sHandle->str[idx].nbFacet] = f;
		sHandle->str[idx].facets[sHandle->str[idx].nbFacet]->globalId = i;
		sHandle->str[idx].nbFacet++;


		if (f->sh.superDest || f->sh.isVolatile) {
			// Link or volatile facet, overides facet settings
			// Must be full opaque and 0 sticking
			// (see SimulationMC.c::PerformBounce)
			//f->sh.isOpaque = TRUE;
			f->sh.opacity = 1.0;
			f->sh.opacity_paramId = -1;
			f->sh.sticking = 0.0;
			f->sh.sticking_paramId = -1;
			if (((f->sh.superDest - 1) >= sHandle->nbSuper || f->sh.superDest < 0)) {
				// Geometry error
				ClearSimulation();
				ReleaseDataport(loader);
				sprintf(err, "Invalid structure (wrong link on F#%d)", i + 1);
				SetErrorSub(err);
				return FALSE;
			}
		}

		// Reset counter in local memory
		//memset(&(f->sh.counter), 0, sizeof(SHHITS));
		f->indices = (int *)malloc(f->sh.nbIndex * sizeof(int));
		buffer += sizeof(SHFACET);
		memcpy(f->indices, buffer, f->sh.nbIndex * sizeof(int));
		buffer += f->sh.nbIndex * sizeof(int);
		f->vertices2 = (Vector2d *)malloc(f->sh.nbIndex * sizeof(Vector2d));
		if (!f->vertices2) {
			SetErrorSub("Not enough memory to load vertices");
			return FALSE;
		}
		memcpy(f->vertices2, buffer, f->sh.nbIndex * sizeof(Vector2d));
		buffer += f->sh.nbIndex * sizeof(Vector2d);
		if (f->sh.useOutgassingFile) {
			f->outgassingMap = (double*)malloc(sizeof(double)*f->sh.outgassingMapWidth*f->sh.outgassingMapHeight);
			if (!f->outgassingMap) {
				SetErrorSub("Not enough memory to load outgassing map");
				return FALSE;
			}
			memcpy(f->outgassingMap, buffer, sizeof(double)*f->sh.outgassingMapWidth*f->sh.outgassingMapHeight);
			buffer += sizeof(double)*f->sh.outgassingMapWidth*f->sh.outgassingMapHeight;
		}

		//Textures
		if (f->sh.isTextured) {
			int nbE = f->sh.texWidth*f->sh.texHeight;
			f->textureSize = nbE * sizeof(AHIT);

			if ((f->hits = (AHIT **)malloc(sizeof(AHIT *)* (1 + sHandle->nbMoments))) == NULL) {
				ReleaseDataport(loader);
				SetErrorSub("Couldn't allocate memory (time moments container, textures)");
				return FALSE;
			}
			memset(f->hits, 0, sizeof(AHIT *)* (1 + sHandle->nbMoments)); //set all pointers to NULL so we can use SAFE_FREE later
			for (size_t m = 0; m < (1 + sHandle->nbMoments); m++) {
				if ((f->hits[m] = (AHIT *)malloc(f->textureSize)) == NULL) { //steady-state plus one for each moment
					ReleaseDataport(loader);
					SetErrorSub("Couldn't allocate memory (textures)");
					return FALSE;
				}
				memset(f->hits[m], 0, f->textureSize);
			}
		}

		//Profiles
		if (f->sh.isProfile) {
			f->profileSize = PROFILE_SIZE * sizeof(APROFILE);
			/*f->profile = (llong *)malloc(f->profileSize);
			memset(f->profile,0,f->profileSize);*/
			if ((f->profile = (APROFILE **)malloc(sizeof(APROFILE *)* (1 + sHandle->nbMoments))) == NULL) {
				ReleaseDataport(loader);
				SetErrorSub("Couldn't allocate memory (time moments container, profiles)");
				return FALSE;
			}
			memset(f->profile, 0, sizeof(APROFILE *)* (1 + sHandle->nbMoments)); //set all pointers to NULL so we can use SAFE_FREE later
			for (size_t m = 0; m < (1 + sHandle->nbMoments); m++) {
				if ((f->profile[m] = (APROFILE *)malloc(f->profileSize)) == NULL) { //steady-state plus one for each moment
					ReleaseDataport(loader);
					SetErrorSub("Couldn't allocate memory (profiles)");
					return FALSE;
				}
				memset(f->profile[m], 0, f->profileSize);
			}
			sHandle->profTotalSize += f->profileSize*(1 + sHandle->nbMoments);
		}

		//Direction
		if (f->sh.countDirection) {
			f->directionSize = f->sh.texWidth*f->sh.texHeight * sizeof(VHIT);
			/*f->direction = (VHIT *)malloc(f->directionSize);
			memset(f->direction,0,f->directionSize);*/
			if ((f->direction = (VHIT **)malloc(sizeof(VHIT *)* (1 + sHandle->nbMoments))) == NULL) {
				ReleaseDataport(loader);
				SetErrorSub("Couldn't allocate memory (time moments container, direction vectors)");
				return FALSE;
			}
			memset(f->direction, 0, sizeof(VHIT *)* (1 + sHandle->nbMoments)); //set all pointers to NULL so we can use SAFE_FREE later
			for (size_t m = 0; m < (1 + sHandle->nbMoments); m++) {
				if ((f->direction[m] = (VHIT *)malloc(f->directionSize)) == NULL) { //steady-state plus one for each moment
					ReleaseDataport(loader);
					SetErrorSub("Couldn't allocate memory (direction vectors)");
					return FALSE;
				}
				memset(f->direction[m], 0, f->directionSize);
			}
			sHandle->dirTotalSize += f->directionSize*(1 + sHandle->nbMoments);
		}

	}


	//Inc values
	buffer = incBuff;
	for (int k = 0; k < sHandle->nbSuper; k++) {
		for (i = 0; i < sHandle->str[k].nbFacet; i++) {
			FACET* f = sHandle->str[k].facets[i];
			if (f->sh.isTextured) {
				int nbE = f->sh.texWidth*f->sh.texHeight;
				f->inc = (double *)malloc(nbE * sizeof(double));
				f->largeEnough = (BOOL *)malloc(sizeof(BOOL)*nbE);
				//f->fullElem = (BOOL *)malloc(sizeof(BOOL)*nbE);
				if (!(f->inc && f->largeEnough /*&& f->fullElem*/)) {
					SetErrorSub("Not enough memory to load");
					return FALSE;
				}
				f->fullSizeInc = 1E30;
				for (j = 0; j < nbE; j++) {
					double incVal = READBUFFER(double);
					/*if (incVal < 0) {
						//f->fullElem[j] = 1;
						f->inc[j] = -incVal;
					}
					else {*/

					//f->fullElem[j] = 0;
					f->inc[j] = incVal;
					/*}*/
					if ((f->inc[j] > 0.0) && (f->inc[j] < f->fullSizeInc)) f->fullSizeInc = f->inc[j];
				}
				for (j = 0; j < nbE; j++) { //second pass, filter out very small cells
					f->largeEnough[j] = (f->inc[j] < ((5.0f)*f->fullSizeInc));
				}
				sHandle->textTotalSize += f->textureSize*(1 + sHandle->nbMoments);


				f->iw = 1.0 / (double)f->sh.texWidthD;
				f->ih = 1.0 / (double)f->sh.texHeightD;
				f->rw = f->sh.U.Norme() * f->iw;
				f->rh = f->sh.V.Norme() * f->ih;
			}
		}
	}


	//CDFs
	size_t size1 = READBUFFER(size_t);
	sHandle->CDFs.reserve(size1);
	for (size_t i = 0; i < size1; i++) {
		std::vector<std::pair<double, double>> newCDF;
		size_t size2 = READBUFFER(size_t);
		newCDF.reserve(size2);
		for (size_t j = 0; j < size2; j++) {
			double valueX = READBUFFER(double);
			double valueY = READBUFFER(double);
			newCDF.push_back(std::make_pair(valueX, valueY));
		}
		sHandle->CDFs.push_back(newCDF);
	}

	//IDs
	size1 = READBUFFER(size_t);
	sHandle->IDs.reserve(size1);
	for (size_t i = 0; i < size1; i++) {
		std::vector<std::pair<double, double>> newID;
		size_t size2 = READBUFFER(size_t);
		newID.reserve(size2);
		for (size_t j = 0; j < size2; j++) {
			double valueX = READBUFFER(double);
			double valueY = READBUFFER(double);
			newID.push_back(std::make_pair(valueX, valueY));
		}
		sHandle->IDs.push_back(newID);
	}



	//Parameters
	size1 = READBUFFER(size_t);
	sHandle->parameters.reserve(size1);
	for (size_t i = 0; i < size1; i++) {
		Parameter newParam = Parameter();
		std::vector<std::pair<double, double>> newValues;
		size_t size2 = READBUFFER(size_t);
		newValues.reserve(size2);
		for (size_t j = 0; j < size2; j++) {
			double valueX = READBUFFER(double);
			double valueY = READBUFFER(double);
			newValues.push_back(std::make_pair(valueX, valueY));

		}
		newParam.SetValues(newValues, FALSE);
		sHandle->parameters.push_back(newParam);
	}

	//Temperatures
	size1 = READBUFFER(size_t);
	sHandle->temperatures.reserve(size1);
	for (size_t i = 0; i < size1; i++) {
		double valueX = READBUFFER(double);
		sHandle->temperatures.push_back(valueX);
	}

	//Time moments
	//size1=READBUFFER(size_t);
	sHandle->moments.reserve(sHandle->nbMoments); //nbMoments already passed
	for (size_t i = 0; i < sHandle->nbMoments; i++) {
		double valueX = READBUFFER(double);
		sHandle->moments.push_back(valueX);
	}

	//Desorption parameter IDs
	size1 = READBUFFER(size_t);
	sHandle->desorptionParameterIDs.reserve(size1);
	for (size_t i = 0; i < size1; i++) {
		size_t valueX = READBUFFER(size_t);
		sHandle->desorptionParameterIDs.push_back(valueX);
	}

	/*//DEBUG

		Parameter *param1=new Parameter();
		param1->AddValue(0.5,0);
		param1->AddValue(0.5001,0.001);
		param1->AddValue(0.6,0.001);
		param1->AddValue(0.8,0);
		sHandle->parameters.push_back(*param1);

		sHandle->str[0].facets[0]->sh.sticking_paramId=0; //set facet 1 as time-dependent desorption*/
	
	//ReleaseDataport(loader); //Commented out as AccessDataport removed

	// Build all AABBTrees
	for (i = 0; i < sHandle->nbSuper; i++)
		sHandle->str[i].aabbTree = BuildAABBTree(sHandle->str[i].facets, sHandle->str[i].nbFacet, 0);

	// Initialise simulation

	seed = GetSeed();
	rseed(seed);
	sHandle->loadOK = TRUE;
	t1 = GetTick();
	printf("  Load %s successful\n", sHandle->name);
	printf("  Geometry: %d vertex %d facets\n", sHandle->nbVertex, sHandle->totalFacet);

	printf("  Geom size: %zd bytes\n", (size_t)(buffer - bufferStart));
	printf("  Number of stucture: %d\n", sHandle->nbSuper);
	printf("  Global Hit: %zd bytes\n", sizeof(SHGHITS));
	printf("  Facet Hit : %zd bytes\n", sHandle->totalFacet * sizeof(SHHITS));
	printf("  Texture   : %zd bytes\n", sHandle->textTotalSize);
	printf("  Profile   : %zd bytes\n", sHandle->profTotalSize);
	printf("  Direction : %zd bytes\n", sHandle->dirTotalSize);

	printf("  Total     : %zd bytes\n", GetHitsSize());
	printf("  Seed: %lu\n", seed);
	printf("  Loading time: %.3f ms\n", (t1 - t0)*1000.0);
	return TRUE;

}

// -------------------------------------------------------

void UpdateHits(Dataport *dpHit, int prIdx, DWORD timeout) {
	switch (sHandle->sMode) {
	case MC_MODE:
	{
		UpdateMCHits(dpHit, prIdx, sHandle->nbMoments, timeout);
	}
		break;
	case AC_MODE:

		UpdateACHits(dpHit, prIdx, timeout);
		break;
	}

}

// -------------------------------------------------------

size_t GetHitsSize() {
	return sHandle->textTotalSize + sHandle->profTotalSize + sHandle->dirTotalSize + sHandle->totalFacet * sizeof(SHHITS) + sizeof(SHGHITS);

}

// -------------------------------------------------------

void ResetTmpCounters() {
	SetState(NULL, "Resetting local cache...", FALSE, TRUE);

	memset(&sHandle->tmpCount, 0, sizeof(SHHITS));

	sHandle->distTraveledSinceUpdate_total = 0.0;
	sHandle->distTraveledSinceUpdate_fullHitsOnly = 0.0;
	sHandle->nbLeakSinceUpdate = 0;
	sHandle->hitCacheSize = 0;
	sHandle->leakCacheSize = 0;

	for (int j = 0; j < sHandle->nbSuper; j++) {
		for (int i = 0; i < sHandle->str[j].nbFacet; i++) {
			FACET *f = sHandle->str[j].facets[i];
			f->ResetCounter();
			f->hitted = FALSE;

			/*
			if (f->hits) {
				for (size_t m = 0; m < (sHandle->nbMoments + 1); m++) {
					memset(f->hits[m], 0, f->textureSize);
				}
			}
			*/
			memset(&(f->hits[0]), 0, f->textureSize*(sHandle->nbMoments + 1));

			/*
			if (f->profile) {
				for (size_t m = 0; m < (sHandle->nbMoments + 1); m++) {
					memset(f->profile[m], 0, f->profileSize);
				}
			}
			*/
			memset(&(f->profile[0]), 0, f->profileSize*(sHandle->nbMoments + 1));

			/*
			if (f->direction) {
				for (size_t m = 0; m < (sHandle->nbMoments + 1); m++) {
					memset(f->direction[m], 0, f->directionSize);
				}
			}
			*/
			memset(&(f->direction[0]), 0, f->directionSize*(sHandle->nbMoments + 1));
		}
	}

}

// -------------------------------------------------------

void ResetSimulation() {
	sHandle->lastHit = NULL;
	sHandle->totalDesorbed = 0;
	ResetTmpCounters();
	if (sHandle->acDensity) memset(sHandle->acDensity, 0, sHandle->nbAC * sizeof(ACFLOAT));

}

// -------------------------------------------------------

BOOL StartSimulation(size_t mode) {

	sHandle->sMode = mode;
	switch (mode) {
	case MC_MODE:
		if (!sHandle->lastHit) StartFromSource();
		return (sHandle->lastHit != NULL);
	case AC_MODE:
		if (sHandle->prgAC != 100) {
			SetErrorSub("AC matrix not calculated");
			return FALSE;
		}
		else {
			sHandle->stepPerSec = 0.0;
			return TRUE;
		}
	}

	SetErrorSub("Unknown simulation mode");
	return FALSE;
}

// -------------------------------------------------------

void RecordHit(const int &type) {
	if (sHandle->hitCacheSize < HITCACHESIZE) {
		sHandle->hitCache[sHandle->hitCacheSize].pos = sHandle->pPos;
		sHandle->hitCache[sHandle->hitCacheSize].type = type;
		sHandle->hitCacheSize++;
	}
}

void RecordLeakPos() {
	// Source region check performed when calling this routine 
	// Record leak for debugging
	RecordHit(HIT_REF);
	RecordHit(LASTHIT);
	if (sHandle->leakCacheSize < LEAKCACHESIZE) {
		sHandle->leakCache[sHandle->leakCacheSize].pos = sHandle->pPos;
		sHandle->leakCache[sHandle->leakCacheSize].dir = sHandle->pDir;
		sHandle->leakCacheSize++;
	}
}

// -------------------------------------------------------

BOOL SimulationRun() {

	// 1s step
	double t0, t1;
	int    nbStep = 1;
	BOOL   goOn;

	if (sHandle->stepPerSec == 0.0) {
		switch (sHandle->sMode) {
		case MC_MODE:

			nbStep = 250;
			break;
		case AC_MODE:
			nbStep = 1;
			break;
		}

	}

	if (sHandle->stepPerSec != 0.0)
		nbStep = (int)(sHandle->stepPerSec + 0.5);
	if (nbStep < 1) nbStep = 1;
	t0 = GetTick();
	switch (sHandle->sMode) {
	case MC_MODE:

		goOn = SimulationMCStep(nbStep);
		break;
	case AC_MODE:
		goOn = SimulationACStep(nbStep);
		break;
	}


	t1 = GetTick();
	sHandle->stepPerSec = (double)(nbStep) / (t1 - t0);
#ifdef _DEBUG
	printf("Running: stepPerSec = %f\n", sHandle->stepPerSec);
#endif

	return !goOn;

}

// -------------------------------------------------------

double GetTick() {

	// Number of sec since the application startup

#ifdef WIN

	if (usePerfCounter) {

		LARGE_INTEGER t, dt;
		QueryPerformanceCounter(&t);
		dt.QuadPart = t.QuadPart - perfTickStart.QuadPart;
		return (double)(dt.QuadPart) / perfTicksPerSec;

	}
	else {


		return (double)((GetTickCount() - tickStart) / 1000.0);

	}

#else

	if (tickStart < 0)
		tickStart = time(NULL);

	struct timeval tv;
	gettimeofday(&tv, NULL);
	return ((double)(tv.tv_sec - tickStart)*1000.0 + (double)tv.tv_usec / 1000.0);

#endif

	}

int GetIDId(int paramId) {

	int i;
	for (i = 0; i < (int)sHandle->desorptionParameterIDs.size() && (paramId != sHandle->desorptionParameterIDs[i]); i++); //check if we already had this parameter Id
	if (i >= (int)sHandle->desorptionParameterIDs.size()) i = -1; //not found
	return i;
}

