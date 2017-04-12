/*
  File:        Worker.h
  Description: Sub processes handling
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

#ifndef _WORKERH_
#define _WORKERH_

#include "Parameter.h"
#include "PugiXML/pugixml.hpp"
#include <string>
#include "GLApp/GLTypes.h"
#include "Vector.h"
#include "Shared.h"
#include "Smp.h"

#define CDF_SIZE 100 //points in a cumulative distribution function

class Geometry;
class MolflowGeometry;
class GLProgress;
class LoadStatus;
class FileReader;

class Worker
{

public:

  // Constructor
  Worker();
  ~Worker();

  // Return a handle to the currently loaded geometry
  Geometry *GetGeometry();
  MolflowGeometry* GetMolflowGeometry();

  // Load a geometry (throws Error)

  void LoadGeometry(char *fileName, BOOL insert=FALSE,BOOL newStr=FALSE); // Load or insert a geometry (throws Error)
  BOOL IsDpInitialized();

  // Inserts a new geometry (throws Error)
  //void InsertGeometry(BOOL newStr,char *fileName);

  // Load a textures(throws Error)
  void LoadTexturesGEO(FileReader *f,int version);
  void RebuildTextures();

  // Save a geometry (throws Error)
  void SaveGeometry(char *fileName,GLProgress *prg,BOOL askConfirm=TRUE,BOOL saveSelected=FALSE,BOOL autoSave=FALSE,BOOL crashSave=FALSE);

  // Save textures (throws Error)

  void ExportTextures(char *fileName,int grouping,int mode,BOOL askConfirm=TRUE,BOOL saveSelected=FALSE);
  void ExportProfiles(char *fileName);

  //Import desorption map
  void ImportDesorption_DES(char *fileName);
  void ImportDesorption_SYN(char *fileName, const size_t &source, const double &time,
	  const size_t &mode, const double &eta0, const double &alpha, const double &cutoffdose,
	  const std::vector<std::pair<double, double>> &convDistr,
	  GLProgress *prg);
  void AnalyzeSYNfile(char *fileName, int *nbFacet, int *nbTextured, int *nbDifferent);
  void ImportCSV(FileReader *file, std::vector<std::vector<std::string>>& table);

  // Return/Set the current filename
  char *GetFileName();
  char *GetShortFileName();
  char *GetShortFileName(char* longFileName);
  void  SetFileName(char *fileName);

  // Set number of processes [1..32] (throws Error)
  void SetProcNumber(int n);

  // Get number of processes
  int GetProcNumber();

  // Set the number of maximum desorption
  void SetMaxDesorption(llong max);

  // Get PID
  DWORD GetPID(int prIdx);

  // Reset simulation
  void ResetStatsAndHits(float appTime);
  void Reload(); //Mark geometry as out of sync with subprocess
  void RealReload(); // Send geometry to subprocess (throws Error)

  // Switch running/stopped
  void StartStop(float appTime,int mode);

    // Switch running/stopped

  void Stop_Public();

  // AC iteration single step
  void StepAC(float appTime);

  // Kill all sub processes
  void KillAll();

  // Get hit counts for sub process
  void Update(float appTime);

  // Send total and facet hit counts to subprocesses
  void SendHits(BOOL skipFacetHits=FALSE);
  void SetLeakCache(LEAK *buffer,size_t *nb,Dataport* dpHit);
  void SetHitCache(HIT *buffer, size_t *nb,Dataport* dpHit);

  // Get process status
  void GetProcStatus(int *states,char **status);

  //Do calculations necessary before launching simulation
  void PrepareToRun();

  //Get ID of parameter name
  int GetParamId(const std::string); 

  // Access to dataport (HIT)
  BYTE *GetHits();
  void  ReleaseHits();

  // Send Compute AC matrix order
  void ComputeAC(float appTime);

  int AddMoment(std::vector<double> newMoments); //Adds a time serie to moments and returns the number of elements
  std::vector<double> ParseMoment(std::string userInput); //Parses a user input and returns a vector of time moments
  void ResetMoments();
  double GetMoleculesPerTP(int moment);

  std::vector<std::pair<double,double>> Generate_ID(int paramId);
  int GenerateNewID(int paramId);
  std::vector<std::pair<double,double>> Generate_CDF(double gasTempKelvins,double gasMassGramsPerMol,size_t size);
  int GenerateNewCDF(double temperature);
  void CalcTotalOutgassing();
  int GetCDFId(double temperature);
int GetIDId(int paramId);

  // Global simulation parameters
  llong  nbAbsorption;      // Total number of molecules absorbed (64 bit integer)
  llong  nbDesorption;      // Total number of molecules generated (64 bit integer)
  llong  nbHit;             // Total number of hit (64 bit integer)


  llong  desorptionLimit;     // Number of desoprtion before halting

  llong  nbLeakTotal;       // Total number of leak
  double distTraveledTotal_total; // Total distance traveled by particles (for mean pumping path calc.)
  double distTraveledTotal_fullHitsOnly; // Total distance traveled by particles between full hits (for mean free path calc.)

  BOOL   running;           // Started/Stopped state
  float  startTime;         // Start time
  float  stopTime;          // Stop time
  float  simuTime;          // Total simulation time
  int    mode;              // Simulation mode
  BOOL   calcAC;            // Calculating AC matrix
  size_t    calcACprg;         // AC matrix progress
  
  std::vector<std::vector<std::pair<double,double>>> CDFs; //cumulative distribution function for each temperature
  std::vector<std::vector<std::pair<double,double>>> IDs; //integrated distribution function for each time-dependent desorption type
  std::vector<double> temperatures; //keeping track of all temperatures that have a CDF already generated
  std::vector<double> moments;             //moments when a time-dependent simulation state is recorded
  std::vector<size_t> desorptionParameterIDs; //time-dependent parameters which are used as desorptions, therefore need to be integrated
  double latestMoment;
  std::vector<std::string> userMoments;    //user-defined text values for defining time moments (can be time or time series)
  //std::vector<unsigned int> testStructures; //structures which are test-cubes
  
  double totalDesorbedMolecules; //Number of molecules desorbed between t=0 and latest_moment
  double finalOutgassingRate; //Number of outgassing molecules / second at latest_moment (constant flow)
  double finalOutgassingRate_Pa_m3_sec; //For the user to see on Global Seetings and in formulas. Not shared with workers
  double gasMass;
  BOOL   enableDecay;
  double halfLife;
  double timeWindowSize;
  BOOL useMaxwellDistribution; //TRUE: Maxwell-Boltzmann distribution, FALSE: All molecules have the same (V_avg) speed
  BOOL calcConstantFlow;

  int motionType;
  Vector3d motionVector1; //base point for rotation
  Vector3d motionVector2; //rotation vector or velocity vector

  BOOL needsReload;

  std::vector<Parameter> parameters;
  BOOL abortRequested; //Signal to stop current operation (Collapse, Analyze, etc.)
  int displayedMoment;
  
	// Current loaded file
  char fullFileName[512];

  // Caches
  HIT  hitCache[HITCACHESIZE];
  LEAK leakCache[HITCACHESIZE];
  size_t    hitCacheSize;
  size_t    leakCacheSize;

private:

  // Process management
  int    nbProcess;
  DWORD  pID[MAX_PROCESS];
  DWORD  pid;
  BOOL   allDone;

  // Geometry handle
  MolflowGeometry *geom;

  // Dataport handles and names
  Dataport *dpControl;
  Dataport *dpHit;
  char      ctrlDpName[32];
  char      loadDpName[32];
  char      hitsDpName[32];

  // Methods
  BOOL ExecuteAndWait(int command, int readyState, int param = 0);
  BOOL Wait(int waitState, LoadStatus *statusWindow = NULL);
  void ResetWorkerStats();
  void ClearHits(BOOL noReload);
  char *GetErrorDetails();
  void ThrowSubProcError(char *message=NULL);
  void Start();
  void Stop();
  void OneStep();
  void InnerStop(float appTime);

};

#endif /* _WORKERH_ */
