/*
  File:        MolFlow.cpp
  Description: Main application class (GUI management)
  Program:     MolFlow+
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
#pragma once

#include <crtdbg.h> //To debug heap corruptions in memory


#include "Interface.h"
#include "Worker.h"
#include "ImportDesorption.h"
#include "TimeSettings.h"
#include "Movement.h"
#include "FacetMesh.h"
#include "FacetDetails.h"
#include "Viewer3DSettings.h"
#include "TextureSettings.h"
#include "GlobalSettings.h"
#include "ProfilePlotter.h"
#include "PressureEvolution.h"
#include "TimewisePlotter.h"
#include "TexturePlotter.h"
#include "OutgassingMap.h"
#include "MomentsEditor.h"
#include "ParameterEditor.h"

class MolFlow : public Interface
{
public:
    MolFlow();
	
	//Public textfields so we can disable them from "Advanced facet parameters":
	GLTextField   *facetFlow;
	GLTextField   *facetFlowArea;
	
    
    void LoadFile(char *fName=NULL);
	void InsertGeometry(BOOL newStr,char *fName=NULL);
	void SaveFile();
    void SaveFileAs();
    
	void ImportDesorption_DES();
	void ExportTextures(int grouping,int mode);
	void ExportProfiles();
    void ClearFacetParams();
    void UpdateFacetParams(BOOL updateSelection=FALSE);
    void ApplyFacetParams();
    
	
    void StartStopSimulation();
	void ResetSimulation(BOOL askConfirm=TRUE);

    void SaveConfig();
    void LoadConfig();
    
	void PlaceComponents();
    void UpdateFacetHits(BOOL allRows=FALSE);
	void QuickPipe();
	BOOL AskToReset(Worker *work=NULL);
	float GetAppTime();
	void ResetAutoSaveTimer();
	BOOL AutoSave(BOOL crashSave=FALSE);
	void CheckForRecovery();
	void ClearParameters();

	//Flow/sticking coeff. conversion
	void calcFlow();
	void calcSticking();

	//char* appName;

	GLButton      *texturePlotterShortcut;
	GLButton      *profilePlotterShortcut;
    //GLButton      *statusSimu;
    
	
    GLTextField   *facetSticking;
    
    
	
    GLCombo       *facetDesType;
	GLTextField   *facetDesTypeN;
    GLCombo       *facetRecType;
	GLLabel       *facetUseDesFileLabel;
	GLLabel       *modeLabel;
	
	GLLabel       *facetPumpingLabel;
	GLTextField   *facetPumping;	
    GLLabel       *facetSLabel;
	
    

    GLLabel       *facetDLabel;
    GLLabel       *facetReLabel;
    GLToggle       *facetFILabel;
	GLToggle      *facetFIAreaLabel;
    //GLLabel       *facetMLabel;
	GLButton      *compACBtn;
	GLButton      *singleACBtn;

	GLButton      *profilePlotterBtn;
	GLButton      *texturePlotterBtn;
	GLButton      *textureScalingBtn;
	GLButton      *globalSettingsBtn;

	GLTitledPanel *inputPanel;
	GLTitledPanel *outputPanel;

    //Dialog
	ImportDesorption *importDesorption;
	TimeSettings     *timeSettings;
	Movement         *movement;
    FacetMesh        *facetMesh;
    FacetDetails     *facetDetails;
	SmartSelection   *smartSelection;
    Viewer3DSettings *viewer3DSettings;
    TextureSettings  *textureSettings;
	GlobalSettings	 *globalSettings;
    ProfilePlotter   *profilePlotter;
	PressureEvolution *pressureEvolution;
	TimewisePlotter  *timewisePlotter;
    TexturePlotter   *texturePlotter;
	OutgassingMap    *outgassingMap;
	MomentsEditor    *momentsEditor;
	ParameterEditor  *parameterEditor;
	char *nbF;

    // Testing
    //int     nbSt;
    //void LogProfile();
    void BuildPipe(double ratio,int steps=0);
    //void BuildPipeStick(double s);
	
	void CrashHandler(Error *e);
	void DoEvents(BOOL forced = FALSE); //Used to catch button presses (check if an abort button was pressed during an operation)

protected:

    int  OneTimeSceneInit();
    int  RestoreDeviceObjects();
	int  InvalidateDeviceObjects();
    int  FrameMove();
    void ProcessMessage(GLComponent *src,int message);
	BOOL EvaluateVariable(VLIST *v, Worker * w, Geometry * geom);

	MolFlow* GetMolflow(){ return this; }
};
