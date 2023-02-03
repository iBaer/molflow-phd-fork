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
#include <cmath>
#include "MolFlow.h"
#include "Facet_shared.h"
#include "MolflowGeometry.h"
#include "File.h"
#include "GLApp/GLMessageBox.h"
#include "GLApp/GLInputBox.h"
#include "NativeFileDialog/molflow_wrapper/nfd_wrapper.h"
#include "GLApp/GLWindowManager.h"
#include "Helper/MathTools.h"
#include <Helper/FormatHelper.h>
#include "ImguiWindow.h"

#include "GLApp/GLMenuBar.h"
#include "GLApp/GLButton.h"
#include "GLApp/GLLabel.h"
#include "GLApp/GLCombo.h"
#include "GLApp/GLTextField.h"

#include "Interface/RecoveryDialog.h"
#include <vector>
#include <string>
#include <numeric> //std::iota
#include <filesystem>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include "direct.h"
#else
#include <unistd.h> //chdir
#endif

// Plotters
#include "Interface/ProfilePlotter.h"
#include "ProfileModes.h"
#include "Interface/PressureEvolution.h"
#include "Interface/TimewisePlotter.h"
#include "Interface/TexturePlotter.h"
#include "ConvergencePlotter.h"

#include "AppUpdater.h"
#include "Worker.h"
#include "Interface/ImportDesorption.h"
#include "Interface/TimeSettings.h"
#include "Interface/Movement.h"
#include "Interface/FacetAdvParams.h"
#include "Interface/FacetDetails.h"
#include "Interface/Viewer3DSettings.h"
#include "Interface/TextureScaling.h"
#include "Interface/GlobalSettings.h"
#include "Interface/OutgassingMapWindow.h"
#include "Interface/MomentsEditor.h"
#include "Interface/FacetCoordinates.h"
#include "Interface/VertexCoordinates.h"
#include "Interface/ParameterEditor.h"
#include "Interface/SmartSelection.h"
#include "Interface/FormulaEditor.h"
#include "Interface/ParticleLogger.h"
#include "Interface/HistogramSettings.h"
#include "Interface/HistogramPlotter.h"
#include "FormulaEvaluator_MF.h"

/*
static const char *fileLFilters = "All MolFlow supported files\0*.txt;*.xml;*.zip;*.geo;*.geo7z;*.syn;*.syn7z;*.str;*.stl;*.ase\0"
"All files\0*.*\0";
static const char *fileInsFilters = "All insertable geometries\0*.txt;*.xml;*.zip;*.geo;*.geo7z;*.syn;*.syn7z;*.stl\0"
"All files\0*.*\0";
const char *fileSFilters = "MolFlow saveable files\0*.xml;*.zip;*.geo;*.geo7z;*.txt\0All files\0*.*\0";
static const char *fileDesFilters = "Desorption files\0*.des\0All files\0*.*\0";
*/

//NativeFileDialog compatible file filters
char fileLoadFilters[] = "txt,xml,zip,geo,syn,str,stl,ase,geo7z,syn7z";
char fileInsertFilters[] = "txt,xml,zip,geo,syn,stl,geo7z,syn7z";
char fileSaveFilters[] = "zip,xml,txt,geo,stl,geo7z";
char fileSelFilters[] = "sel";
char fileTexFilters[] = "txt";
char fileProfFilters[] = "csv,txt";


int cSize = 4;
int   cWidth[] = { 30, 56, 50, 50 };
const char* cName[] = { "#", "Hits", "Des", "Abs" };

std::vector<std::string> formulaPrefixes = { "A","D","H","MCH","P","DEN","Z","V","T","AR","a","d","h","mch","p","den","z","v","t","ar","," };
char formulaSyntax[] =
R"(MC Variables: An (Absorption on facet n), Dn (Desorption on facet n), Hn (Hit on facet n)
Pn (Pressure [mbar] on facet n), DENn (Density [1/m3] on facet n)
Zn (Imp. rate on facet n), Vn (avg. speed [m/s] on facet n), Tn (temp[K] of facet n)
SUMABS (total absorbed), SUMDES (total desorbed), SUMHIT (total hit)

SUM(H,3,8)    calculates the sum of hits on facets 3,4,... ...7,8. (Works with H,A,D,AR).
AVG(P,3,8)    calculates the average pressure (area-wise) on facets 3 to 8 (Works with P,D,Z)
SUM(H,S3)    calculates the sum of hits on selection group 3 (works with H,A,D,AR)
AVG(DEN,S2) calculates the average (area-wise) on facets belonging to sel. group 2
SUM(H,SEL)    calculates the sum of hits on the current selection. (Works with H,A,D,AR)
AVG(Z,SEL)    calculates the average impingement rate on the current selection
For the last two, might need to manually refresh formulas after you change the selection.

Area variables: ARn (Area of facet n), DESAR (total desorption area), ABSAR (total absorption area)

Final (constant) outgassing rate [mbar*l/s]: QCONST
Final (constant) outgassing rate [molecules/s]: QCONST_N
Total desorbed molecules until last moment: [molecules]: NTOT
Gas mass [g/mol]: GASMASS

Mean Pumping Path: MPP (average path of molecules in the system before absorption)
Mean Free Path:      MFP (average path of molecules between two wall hits)

Math functions: sin(), cos(), tan(), sinh(), cosh(), tanh(), asin(), acos(),
                     atan(), exp(), ln(), pow(x,y), log2(), log10(), inv(), sqrt(), abs()

Constants:  Kb (Boltzmann's constant), R (Gas constant), Na (Avogadro's number), PI
)";
int formulaSyntaxHeight = 380;

MolFlow *mApp;

//Menu elements, Molflow specific:
//#define MENU_FILE_IMPORTDES_DES 140
#define MENU_FILE_IMPORTDES_SYN 141

#define MENU_FILE_EXPORTTEXTURE_AREA 151
#define MENU_FILE_EXPORTTEXTURE_MCHITS 152
#define MENU_FILE_EXPORTTEXTURE_IMPINGEMENT 153
#define MENU_FILE_EXPORTTEXTURE_PART_DENSITY 154
#define MENU_FILE_EXPORTTEXTURE_GAS_DENSITY 155
#define MENU_FILE_EXPORTTEXTURE_PRESSURE 156
#define MENU_FILE_EXPORTTEXTURE_AVG_V 157
#define MENU_FILE_EXPORTTEXTURE_V_VECTOR 158
#define MENU_FILE_EXPORTTEXTURE_N_VECTORS 159

#define MENU_FILE_EXPORTTEXTURE_AREA_COORD 171
#define MENU_FILE_EXPORTTEXTURE_MCHITS_COORD  172
#define MENU_FILE_EXPORTTEXTURE_IMPINGEMENT_COORD  173
#define MENU_FILE_EXPORTTEXTURE_PART_DENSITY_COORD  174
#define MENU_FILE_EXPORTTEXTURE_GAS_DENSITY_COORD  175
#define MENU_FILE_EXPORTTEXTURE_PRESSURE_COORD  176
#define MENU_FILE_EXPORTTEXTURE_AVG_V_COORD  177
#define MENU_FILE_EXPORTTEXTURE_V_VECTOR_COORD  178
#define MENU_FILE_EXPORTTEXTURE_N_VECTORS_COORD  179

#define MENU_TOOLS_MOVINGPARTS 410

#define MENU_SELECT_HASDESFILE 361
#define MENU_FACET_OUTGASSINGMAP 362

#define MENU_TIME_SETTINGS          900
#define MENU_TIMEWISE_PLOTTER       901
#define MENU_TIME_PRESSUREEVOLUTION 902
#define MENU_TIME_MOMENTS_EDITOR    903
#define MENU_TIME_PARAMETER_EDITOR  904

// Name: WinMain()
// Desc: Entry point to the program. Initializes everything, and goes into a
//       message-processing loop. Idle time is used to render the scene.

//INT WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, INT) //Can be replaced with main() if including SDL2Main.lib
int main(int argc, char* argv[])
{
	mApp = new MolFlow();

#ifndef _WIN32
	//Change working directory to executable path (if launched by dbl-click)
	std::string myPath = FileUtils::GetPath(argv[0]);
	if (!myPath.empty()) chdir(myPath.c_str());
#endif

	if (!mApp->Create(1024, 800, false)) {
		char *logs = GLToolkit::GetLogs();
#ifdef _WIN32
		if (logs) MessageBox(nullptr, logs, "Molflow [Fatal error]", MB_OK);
#else
		if (logs) {
			Log::console_error("Molflow [Fatal error]\n");
            Log::console_error("{}", logs);
		}
#endif
		SAFE_FREE(logs);
		delete mApp;
		return -1;
    }
	try {
		mApp->Run();
	}
	catch(const std::exception &e) {
		mApp->CrashHandler(e);
	}
	delete mApp;
	return 0;
}

// Name: MolFlow()
// Desc: Application constructor. Sets default attributes for the app.

MolFlow::MolFlow()
{
	mApp = this; //to refer to the app as extern variable

	//Different Molflow implementation:
	facetAdvParams = nullptr;
	facetDetails = nullptr;
	viewer3DSettings = nullptr;
	textureScaling = nullptr;
	formulaEditor = nullptr;
	globalSettings = nullptr;
	profilePlotter = nullptr;
    texturePlotter = nullptr;

	//Molflow only:
	movement = nullptr;
	timewisePlotter = nullptr;
	pressureEvolution = nullptr;
    outgassingMapWindow = nullptr;
	momentsEditor = nullptr;
	parameterEditor = nullptr;
	importDesorption = nullptr;
	timeSettings = nullptr;

	useOldXMLFormat = false;
	FormulaEvaluator* eval = new FormulaEvaluator_MF(&worker,(MolflowGeometry*)worker.GetGeometry(),&selections);
    formula_ptr = std::make_shared<Formulas>(eval);
}

// Name: OneTimeSceneInit()
// Desc: Called during initial app startup, this function performs all the
//       permanent initialization.
int MolFlow::OneTimeSceneInit()
{
	/*
	//Enable memory check at EVERY malloc/free operation:
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
	*/

	OneTimeSceneInit_shared_pre();

	menu->GetSubMenu("File")->Add("Export selected textures");
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->Add("Facet by facet");
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Cell Area (cm\262)", MENU_FILE_EXPORTTEXTURE_AREA);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("# of MC Hits", MENU_FILE_EXPORTTEXTURE_MCHITS);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Impingement rate (1/s/m\262)", MENU_FILE_EXPORTTEXTURE_IMPINGEMENT);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Particle density (1/m\263)", MENU_FILE_EXPORTTEXTURE_PART_DENSITY);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Gas density (kg/m\263)", MENU_FILE_EXPORTTEXTURE_GAS_DENSITY);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Pressure (mbar)", MENU_FILE_EXPORTTEXTURE_PRESSURE);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Avg. Velocity (m/s)", MENU_FILE_EXPORTTEXTURE_AVG_V);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("Velocity vector (m/s)", MENU_FILE_EXPORTTEXTURE_V_VECTOR);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("Facet by facet")->Add("# of velocity vectors", MENU_FILE_EXPORTTEXTURE_N_VECTORS);

	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->Add("By X,Y,Z coordinates");
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Cell Area (cm\262)", MENU_FILE_EXPORTTEXTURE_AREA_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("# of MC Hits", MENU_FILE_EXPORTTEXTURE_MCHITS_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Impingement rate (1/s/m\262)", MENU_FILE_EXPORTTEXTURE_IMPINGEMENT_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Particle density (1/m\263)", MENU_FILE_EXPORTTEXTURE_PART_DENSITY_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Gas density (kg/m\263)", MENU_FILE_EXPORTTEXTURE_GAS_DENSITY_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Pressure (mbar)", MENU_FILE_EXPORTTEXTURE_PRESSURE_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Avg. Velocity (m/s)", MENU_FILE_EXPORTTEXTURE_AVG_V_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("Velocity vector (m/s)", MENU_FILE_EXPORTTEXTURE_V_VECTOR_COORD);
	menu->GetSubMenu("File")->GetSubMenu("Export selected textures")->GetSubMenu("By X,Y,Z coordinates")->Add("# of velocity vectors", MENU_FILE_EXPORTTEXTURE_N_VECTORS_COORD);

	menu->GetSubMenu("File")->Add("Import desorption from SYN file",MENU_FILE_IMPORTDES_SYN);
	//menu->GetSubMenu("File")->GetSubMenu("Import desorption file")->Add("SYN file", );
	//menu->GetSubMenu("File")->GetSubMenu("Import desorption file")->Add("DES file (deprecated)", MENU_FILE_IMPORTDES_DES);

	menu->GetSubMenu("File")->Add(nullptr); // Separator
	menu->GetSubMenu("File")->Add("E&xit", MENU_FILE_EXIT); //Moved here from OnetimeSceneinit_shared to assert it's the last menu item

	menu->GetSubMenu("Selection")->Add(nullptr); // Separator
	menu->GetSubMenu("Selection")->Add("Select Desorption", MENU_FACET_SELECTDES);
	menu->GetSubMenu("Selection")->Add("Select Dyn. Desorption (Outg.Map)", MENU_SELECT_HASDESFILE);
	menu->GetSubMenu("Selection")->Add("Select Reflective", MENU_FACET_SELECTREFL);
	menu->GetSubMenu("Selection")->Add("Select volatile facets", MENU_FACET_SELECTVOL);

	menu->GetSubMenu("Tools")->Add(nullptr);
	menu->GetSubMenu("Tools")->Add("Moving parts...", MENU_TOOLS_MOVINGPARTS);
	menu->GetSubMenu("Facet")->Add("Convert to outgassing map...", MENU_FACET_OUTGASSINGMAP);

	menu->Add("Time");
	menu->GetSubMenu("Time")->Add("Time settings...", MENU_TIME_SETTINGS, SDLK_i, ALT_MODIFIER);
	menu->GetSubMenu("Time")->Add("Edit moments...", MENU_TIME_MOMENTS_EDITOR);
	menu->GetSubMenu("Time")->Add("Edit parameters...", MENU_TIME_PARAMETER_EDITOR);
	menu->GetSubMenu("Time")->Add(nullptr);
	menu->GetSubMenu("Time")->Add("Timewise plotter", MENU_TIMEWISE_PLOTTER);
	menu->GetSubMenu("Time")->Add("Pressure evolution", MENU_TIME_PRESSUREEVOLUTION);

	viewerMoreButton = new GLButton(0, "<< View");
	togglePanel->Add(viewerMoreButton);

	shortcutPanel = new GLTitledPanel("Shortcuts");
	shortcutPanel->SetClosable(true);
	shortcutPanel->Close();
	Add(shortcutPanel);

	profilePlotterBtn = new GLButton(0, "Profile pl.");
	shortcutPanel->Add(profilePlotterBtn);

	texturePlotterBtn = new GLButton(0, "Texture pl.");
	shortcutPanel->Add(texturePlotterBtn);

	textureScalingBtn = new GLButton(0, "Tex.scaling");
	shortcutPanel->Add(textureScalingBtn);

	/*globalSettingsBtn = new GLButton(0, "<< Sim");
	simuPanel->Add(globalSettingsBtn);
*/
	/*
	statusSimu = new GLButton(0,"...");
	simuPanel->Add(statusSimu);
	*/

	inputPanel = new GLTitledPanel("Particles in");
	facetPanel->Add(inputPanel);

	facetDLabel = new GLLabel("Desorption");
	facetPanel->Add(facetDLabel);
	facetDesType = new GLCombo(0);
	facetDesType->SetSize(5);
	facetDesType->SetValueAt(0, "None");
	facetDesType->SetValueAt(1, "Uniform");
	facetDesType->SetValueAt(2, "Cosine");
	facetDesType->SetValueAt(3, "Cosine^N");
	facetDesType->SetValueAt(4, "Recorded");
	inputPanel->Add(facetDesType);

	facetDesTypeN = new GLTextField(0, nullptr);
	facetDesTypeN->SetEditable(false);
	facetPanel->Add(facetDesTypeN);

	facetFILabel = new GLToggle(0, "Outgassing (mbar*l/s):");
	facetFILabel->SetEnabled(false);
	facetFILabel->SetState(true);
	inputPanel->Add(facetFILabel);
	facetFlow = new GLTextField(0, nullptr);
	inputPanel->Add(facetFlow);

	facetFIAreaLabel = new GLToggle(1, "Outg/area(mbar*l/s/cm\262):");
	facetFIAreaLabel->SetEnabled(false);
	inputPanel->Add(facetFIAreaLabel);
	facetFlowArea = new GLTextField(0, nullptr);
	inputPanel->Add(facetFlowArea);

	outputPanel = new GLTitledPanel("Particles out");
	facetPanel->Add(outputPanel);

	facetSLabel = new GLLabel("Sticking factor:");
	outputPanel->Add(facetSLabel);
	facetSticking = new GLTextField(0, nullptr);
	outputPanel->Add(facetSticking);

	facetPumpingLabel = new GLLabel("Pumping Speed (l/s):");
	outputPanel->Add(facetPumpingLabel);
	facetPumping = new GLTextField(0, nullptr);
	outputPanel->Add(facetPumping);

	facetTempLabel = new GLLabel("Temperature (\260K):");
	facetPanel->Add(facetTempLabel);
	facetTemperature = new GLTextField(0, nullptr);
	facetPanel->Add(facetTemperature);



	facetReLabel = new GLLabel("Profile:");
	facetPanel->Add(facetReLabel);
	facetProfileCombo = new GLCombo(0);
	size_t nbRecModes=(size_t)ProfileRecordModes::NUMITEMS;
	facetProfileCombo->SetSize(nbRecModes);
	for (size_t i = 0; i < nbRecModes; i++) {
		facetProfileCombo->SetValueAt(i, profileRecordModeDescriptions[(ProfileRecordModes)i].first.c_str()); //long description
	}
	facetPanel->Add(facetProfileCombo);

	facetAdvParamsBtn = new GLButton(0, "<< Adv");
	facetPanel->Add(facetAdvParamsBtn);

	facetList = new GLList(0);
	facetList->SetWorker(&worker);
	facetList->SetGrid(true);
	facetList->SetSelectionMode(MULTIPLE_ROW);
	facetList->SetSize(4, 0);
	facetList->SetColumnWidths((int*)cWidth);
	facetList->SetColumnLabels((const char**)cName);
	facetList->SetColumnLabelVisible(true);
	facetList->Sortable = true;
	Add(facetList);

	facetAdvParams = new FacetAdvParams(&worker); //To use its UpdatefacetParams() routines

    Parameter::LoadParameterCatalog(worker.parameters);
	OneTimeSceneInit_shared_post();
	
	return GL_OK;
}

void MolFlow::PlaceComponents() {

	int sx = m_screenWidth - 205;
	int sy = 30;

	Place3DViewer();

	geomNumber->SetBounds(sx, 3, 202, 18);

	// Viewer settings ----------------------------------------
	togglePanel->SetBounds(sx, sy, 202, 112);

	togglePanel->SetCompBounds(showRule, 5, 20, 60, 18);
	togglePanel->SetCompBounds(showNormal, 70, 20, 60, 18);
	togglePanel->SetCompBounds(showUV, 135, 20, 60, 18);

	togglePanel->SetCompBounds(showLine, 5, 42, 60, 18);
	togglePanel->SetCompBounds(showLeak, 70, 42, 60, 18);
	togglePanel->SetCompBounds(showHit, 135, 42, 60, 18);

	togglePanel->SetCompBounds(showVolume, 5, 64, 60, 18);
	togglePanel->SetCompBounds(showTexture, 70, 64, 60, 18);
    togglePanel->SetCompBounds(showFacetId, 135, 64, 60, 18);

	togglePanel->SetCompBounds(viewerMoreButton, 5, 86, 55, 18);
	togglePanel->SetCompBounds(showIndex, 70, 86, 60, 18);
	togglePanel->SetCompBounds(showVertexId, 135, 86, 60, 18);

	sy += (togglePanel->GetHeight() + 5);

	// Selected facet -----------------------------------------
	facetPanel->SetBounds(sx, sy, 202, 330);

	facetPanel->SetCompBounds(inputPanel, 5, 16, 192, 90);

	int cursorY = 15;
	inputPanel->SetCompBounds(facetDLabel, 5, cursorY, 60, 18);
	inputPanel->SetCompBounds(facetDesType, 65, cursorY, 80, 18);
	inputPanel->SetCompBounds(facetDesTypeN, 150, cursorY, 30, 18);

	inputPanel->SetCompBounds(facetFILabel, 5, cursorY += 25, 110, 18);
	inputPanel->SetCompBounds(facetFlow, 140, cursorY, 45, 18);

	inputPanel->SetCompBounds(facetFIAreaLabel, 5, cursorY += 25, 110, 18);
	inputPanel->SetCompBounds(facetFlowArea, 140, cursorY, 45, 18);

	facetPanel->SetCompBounds(outputPanel, 5, cursorY += 45, 192, 65);

	outputPanel->SetCompBounds(facetSLabel, 7, cursorY = 15, 100, 18);
	outputPanel->SetCompBounds(facetSticking, 140, cursorY, 45, 18);

	outputPanel->SetCompBounds(facetPumpingLabel, 7, cursorY += 25, 100, 18);
	outputPanel->SetCompBounds(facetPumping, 140, cursorY, 45, 18);

	facetPanel->SetCompBounds(facetSideLabel, 7, cursorY = 180, 50, 18);
	facetPanel->SetCompBounds(facetSideType, 65, cursorY, 130, 18);

	facetPanel->SetCompBounds(facetTLabel, 7, cursorY += 25, 100, 18);
	facetPanel->SetCompBounds(facetOpacity, 110, cursorY, 82, 18);

	facetPanel->SetCompBounds(facetTempLabel, 7, cursorY += 25, 100, 18);
	facetPanel->SetCompBounds(facetTemperature, 110, cursorY, 82, 18);

	facetPanel->SetCompBounds(facetAreaLabel, 7, cursorY += 25, 100, 18);
	facetPanel->SetCompBounds(facetArea, 110, cursorY, 82, 18);

	facetPanel->SetCompBounds(facetReLabel, 7, cursorY += 25, 60, 18);
	facetPanel->SetCompBounds(facetProfileCombo, 65, cursorY, 130, 18);

	facetPanel->SetCompBounds(facetAdvParamsBtn, 5, cursorY += 25, 48, 18);
	facetPanel->SetCompBounds(facetDetailsBtn, 56, cursorY, 45, 18);
	facetPanel->SetCompBounds(facetCoordBtn, 104, cursorY, 45, 18);
	facetPanel->SetCompBounds(facetApplyBtn, 153, cursorY, 44, 18);

	sy += facetPanel->GetHeight() + 5;

	shortcutPanel->SetBounds(sx, sy, 202, 40);
	shortcutPanel->SetCompBounds(profilePlotterBtn, 5, 15, 60, 18);
	shortcutPanel->SetCompBounds(texturePlotterBtn, 70, 15, 60, 18);
	shortcutPanel->SetCompBounds(textureScalingBtn, 135, 15, 60, 18);

	sy += shortcutPanel->GetHeight() + 5;

	// Simulation ---------------------------------------------
	simuPanel->SetBounds(sx, sy, 202, 169);

	simuPanel->SetCompBounds(globalSettingsBtn, 5, 20, 48, 19);
	simuPanel->SetCompBounds(startSimu, 58, 20, 66, 19);
	simuPanel->SetCompBounds(resetSimu, 128, 20, 66, 19);
	//simuPanel->SetCompBounds(statusSimu,175,20,20,19);
	//simuPanel->SetCompBounds(modeLabel, 5, 45, 30, 18);
	//simuPanel->SetCompBounds(compACBtn, 130, 45, 65, 19);
	simuPanel->SetCompBounds(autoFrameMoveToggle, 5, 45, 65, 19);
	simuPanel->SetCompBounds(forceFrameMoveButton, 128, 45, 66, 19);

	//simuPanel->SetCompBounds(compACBtn,123,45,52,19);
	//simuPanel->SetCompBounds(singleACBtn,178,45,20,19);
	simuPanel->SetCompBounds(hitLabel, 5, 70, 30, 18);
	simuPanel->SetCompBounds(hitNumber, 40, 70, 155, 18);
	simuPanel->SetCompBounds(desLabel, 5, 95, 30, 18);
	simuPanel->SetCompBounds(desNumber, 40, 95, 155, 18);
	simuPanel->SetCompBounds(leakLabel, 5, 120, 30, 18);
	simuPanel->SetCompBounds(leakNumber, 40, 120, 155, 18);
	simuPanel->SetCompBounds(sTimeLabel, 5, 145, 30, 18);

	simuPanel->SetCompBounds(sTime, 40, 145, 155, 18);

	sy += (simuPanel->GetHeight() + 5);

	
	int lg = m_screenHeight -23 /*- (nbFormula * 25)*/;

	facetList->SetBounds(sx, sy, 202, lg - sy);

	/*
	for (int i = 0; i < nbFormula; i++) {
		formulas[i].name->SetBounds(sx, lg + 5, 95, 18);
		formulas[i].value->SetBounds(sx + 90, lg + 5, 87, 18);
		formulas[i].setBtn->SetBounds(sx + 182, lg + 5, 20, 18);
		lg += 25;
	}
	*/
}

// Name: ClearFacetParams()
// Desc: Reset selected facet parameters.

void MolFlow::ClearFacetParams() {
	facetPanel->SetTitle("Selected Facet (none)");
	facetSticking->Clear();
	facetSticking->SetEditable(false);
	facetFILabel->SetEnabled(false);
	facetFIAreaLabel->SetEnabled(false);
	facetFlow->Clear();
	facetFlow->SetEditable(false);
	facetFlowArea->Clear();
	facetFlowArea->SetEditable(false);
	facetArea->SetEditable(false);
	facetArea->Clear();
	facetPumping->SetEditable(false);
	facetPumping->Clear();
	facetOpacity->Clear();
	facetOpacity->SetEditable(false);
	facetTemperature->Clear();
	facetTemperature->SetEditable(false);
	facetSideType->SetSelectedValue("");
	facetSideType->SetEditable(false);
	facetDesType->SetSelectedValue("");
	facetDesType->SetEditable(false);
	facetDesTypeN->SetText("");
	facetDesTypeN->SetEditable(false);
	facetProfileCombo->SetSelectedValue("");
	facetProfileCombo->SetEditable(false);
}

// Name: ApplyFacetParams()
// Desc: Apply facet parameters.

void MolFlow::ApplyFacetParams() {

	Geometry *geom = worker.GetGeometry();
	size_t nbFacet = geom->GetNbFacet();

	// Sticking
	double sticking;
	bool stickingNotNumber;
	bool doSticking = false;
	if (facetSticking->GetNumber(&sticking)) {
		if (sticking<0.0 || sticking>1.0) {
			GLMessageBox::Display("Sticking must be in the range [0,1]", "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
		doSticking = true;
		stickingNotNumber = false;
	}
	else {
		if (facetSticking->GetText() == "...") doSticking = false;
		else {/*
			GLMessageBox::Display("Invalid sticking number","Error",GLDLG_OK,GLDLG_ICONERROR);
			UpdateFacetParams();
			return;*/
			doSticking = true;
			stickingNotNumber = true;
		}
	}

	// opacity
	double opacity;
	bool doOpacity = false;
	bool opacityNotNumber;
	if (facetOpacity->GetNumber(&opacity)) {
		if (opacity<0.0 || opacity>1.0) {
			GLMessageBox::Display("Opacity must be in the range [0,1]", "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
		doOpacity = true;
		opacityNotNumber = false;
	}
	else {
		if (facetOpacity->GetText() == "...") doOpacity = false;
		else {/*
			GLMessageBox::Display("Invalid opacity number","Error",GLDLG_OK,GLDLG_ICONERROR);
			UpdateFacetParams();
			return;*/
			doOpacity = true;
			opacityNotNumber = true;
		}
	}

	// temperature
	double temperature;
	bool doTemperature = false;
	if (facetTemperature->GetNumber(&temperature)) {
		if (temperature < 0.0) {
			GLMessageBox::Display("Temperature must be positive", "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
		doTemperature = true;
	}
	else {
		if (facetTemperature->GetText() == "...") doTemperature = false;
		else {
			GLMessageBox::Display("Invalid temperature number", "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
	}

	// Outgassing
	double outgassing = 0.0;
	bool doFlow = false;
	bool outgassingNotNumber;
	//Calculate outgassing
	if (facetFILabel->GetState() && facetFlow->GetText() != "..." && facetDesType->GetSelectedIndex() != 0
		&& facetDesType->GetSelectedValue() != "..." && facetFlow->IsEditable()) {  //We want outgassing
		if (facetFlow->GetNumber(&outgassing)) { //If we can parse the number
			if (outgassing <= 0.0) {
				GLMessageBox::Display("Outgassing must be positive", "Error", GLDLG_OK, GLDLG_ICONERROR);
				return;
			}
			doFlow = true;
			outgassingNotNumber = false;
		}
		else { //could not parse as number
			doFlow = true;
			outgassingNotNumber = true;
		}
	}

	// Outgassing per area
	double flowA = 0;
	bool doFlowA = false;
	//Calculate outgassing

	if (facetFIAreaLabel->GetState() && facetFlowArea->GetText()!= "..."
		&& facetDesType->GetSelectedIndex() != 0 && facetDesType->GetSelectedValue() != "..." && facetFlowArea->IsEditable()) { //We want outgassing per area
		if (facetFlowArea->GetNumber(&flowA)) { //Can be parsed as number
			if (flowA <= 0.0) {
				GLMessageBox::Display("Outgassing per area must be positive", "Error", GLDLG_OK, GLDLG_ICONERROR);
				return;
			}
			doFlowA = true;
		}
		else {
			GLMessageBox::Display("Invalid outgassing per area number", "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
	}

	// Desorption type
	int desorbType = facetDesType->GetSelectedIndex();

	double desorbTypeN;
	bool doDesorbTypeN = false;
	if (desorbType == 3) {
		if (facetDesTypeN->GetNumber(&desorbTypeN)) {
			if (desorbTypeN <= 0.0) {
				GLMessageBox::Display("Desorption type exponent must be greater than 0.0", "Error", GLDLG_OK, GLDLG_ICONERROR);
				UpdateFacetParams(false);
				return;
			}
			doDesorbTypeN = true;
		}
		else {
			if (facetDesTypeN->GetText() == "...") doDesorbTypeN = false;
			else {
				GLMessageBox::Display("Invalid desorption type exponent", "Error", GLDLG_OK, GLDLG_ICONERROR);
				return;
			}
		}
	}

	// Record (profile) type
	int rType = facetProfileCombo->GetSelectedIndex(); // -1 if "..."

	// 2sided
	int is2Sided = facetSideType->GetSelectedIndex();

    if(desorbType == DES_ANGLEMAP && facetAdvParams->IsAngleMapRecording()){
        char errMsg[512];
        sprintf(errMsg, "Angle map can't be used for DESORPTION and RECORDING\n");
        GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
        return;
    }

	//Check complete, let's apply
	if (facetAdvParams && facetAdvParams->IsVisible()) {
		if (!facetAdvParams->Apply()) {
			return;
		}
	}

	if (!AskToReset()) return;
	
	changedSinceSave = true;

	// Update facets (local)
	for (int i = 0; i < nbFacet; i++) {
		InterfaceFacet *f = geom->GetFacet(i);
		if (f->selected) {
			if (doSticking) {
				if (!stickingNotNumber) {
					f->sh.sticking = sticking;
					f->userSticking = "";
				}
				else {
					f->userSticking = facetSticking->GetText();
				}
			}

			if (doOpacity) {
				if (!opacityNotNumber) {
					f->sh.opacity = opacity;
					f->userOpacity = "";
				}
				else {
					f->userOpacity = facetOpacity->GetText();
				}
			}
			if (doTemperature) f->sh.temperature = temperature;
			if (doFlow) {
				if (!outgassingNotNumber) {
					f->sh.outgassing = outgassing*MBARLS_TO_PAM3S; //0.1: mbar*l/s -> Pa*m3/s
					f->userOutgassing = "";
				}
				else {
					f->userOutgassing = facetFlow->GetText();
				}
			}
			if (doFlowA/* && !useMapA*/) f->sh.outgassing = flowA*f->GetArea()*MBARLS_TO_PAM3S;
			if (desorbType >= 0) {
				if (desorbType == 0) f->sh.outgassing = 0.0;
				if (desorbType != 3) f->sh.desorbTypeN = 0.0;
				f->sh.desorbType = desorbType;
				if (doDesorbTypeN) f->sh.desorbTypeN = desorbTypeN;
			}

			if (rType >= 0) {
				f->sh.profileType = rType;
				//f->wp.isProfile = (rType!=PROFILE_NONE); //included below by f->UpdateFlags();
			}
			if (is2Sided >= 0) f->sh.is2sided = is2Sided;

			f->sh.maxSpeed = 4.0 * sqrt(2.0*8.31*f->sh.temperature / 0.001 / worker.model->wp.gasMass);
			f->UpdateFlags();
		}
	}

	// Mark "needsReload" to sync changes with workers on next simulation start
	worker.Reload();

	worker.CalcTotalOutgassing();
	UpdateFacetParams(false);
	if (profilePlotter) profilePlotter->Refresh();
	if (pressureEvolution) pressureEvolution->Refresh();
	if (timewisePlotter) timewisePlotter->Refresh();
	//if (facetAdvParams) facetAdvParams->Refresh();
}

// Name: UpdateFacetParams()
// Desc: Update selected facet parameters.

void MolFlow::UpdateFacetParams(bool updateSelection) { //Calls facetAdvParams->Refresh()

	char tmp[256];

	// Update params
	Geometry *geom = worker.GetGeometry();
	// Get list of selected facet
	std::vector<size_t> selectedFacets = geom->GetSelectedFacets();
	size_t nbSel = selectedFacets.size();

	if (nbSel > 0) {

		InterfaceFacet *f0;
		InterfaceFacet *f;

		

		f0 = geom->GetFacet(selectedFacets[0]);

		double f0Area = f0->GetArea();
		double sumArea = f0Area; //sum facet area

		bool stickingE = true;
		bool opacityE = true;
		bool temperatureE = true;
		bool flowE = true;
		bool flowAreaE = true;
		bool desorbTypeE = true;
		bool desorbTypeNE = true;
		bool recordE = true;
		bool is2sidedE = true;

		for (size_t sel = 1; sel < selectedFacets.size();sel++) {
			f = geom->GetFacet(selectedFacets[sel]);
			double fArea = f->GetArea();
			stickingE = stickingE && (f0->userSticking == f->userSticking) && IsEqual(f0->sh.sticking, f->sh.sticking);
			opacityE = opacityE && (f0->userOpacity == f->userOpacity) && IsEqual(f0->sh.opacity, f->sh.opacity);
			temperatureE = temperatureE && IsEqual(f0->sh.temperature, f->sh.temperature);
			flowE = flowE && f0->userOutgassing == f->userOutgassing && IsEqual(f0->sh.outgassing, f->sh.outgassing);
			flowAreaE = flowAreaE && IsEqual(f0->sh.outgassing / f0Area, f->sh.outgassing / fArea);
			is2sidedE = is2sidedE && (f0->sh.is2sided == f->sh.is2sided);
			desorbTypeE = desorbTypeE && (f0->sh.desorbType == f->sh.desorbType);
			desorbTypeNE = desorbTypeNE && IsEqual(f0->sh.desorbTypeN, f->sh.desorbTypeN);
			recordE = recordE && (f0->sh.profileType == f->sh.profileType);  //profiles
			sumArea += fArea;
		}

		if (nbSel == 1)
			sprintf(tmp, "Selected Facet (#%zd)", selectedFacets[0] + 1);
		else
			sprintf(tmp, "Selected Facet (%zd selected)", selectedFacets.size());

		// Old STR compatibility
		//if (stickingE && f0->wp.superDest) stickingE = false;

		facetPanel->SetTitle(tmp);
		if (selectedFacets.size() > 1) facetAreaLabel->SetText("Sum Area (cm\262):");
		else facetAreaLabel->SetText("Area (cm\262):");
		facetArea->SetText(sumArea);
		if (stickingE) {
			if (f0->userSticking.length() == 0)
				facetSticking->SetText(f0->sh.sticking);
			else facetSticking->SetText(f0->userSticking.c_str());
		}
		else facetSticking->SetText("...");

		if (opacityE) {
			if (f0->userOpacity.length() == 0)
				facetOpacity->SetText(f0->sh.opacity);
			else facetOpacity->SetText(f0->userOpacity.c_str());
		}
		else facetOpacity->SetText("...");

		if (temperatureE) facetTemperature->SetText(f0->sh.temperature); else facetTemperature->SetText("...");
		if (is2sidedE) facetSideType->SetSelectedIndex(f0->sh.is2sided); else facetSideType->SetSelectedValue("...");
		if (desorbTypeNE) facetDesTypeN->SetText(f0->sh.desorbTypeN); else facetDesTypeN->SetText("...");
		if (recordE) facetProfileCombo->SetSelectedIndex(f0->sh.profileType); else facetProfileCombo->SetSelectedValue("...");

		if (selectedFacets.size() == 1) {
			facetPumping->SetEditable(true);
			calcFlow();
		}
		else {
			facetPumping->SetEditable(false);
			facetPumping->SetText("...");
		}

		if (desorbTypeE) {
			facetDesType->SetSelectedIndex(f0->sh.desorbType);
			if (f0->sh.desorbType > DES_NONE) { //There is some desorption

				//facetFlow->SetEnabled(true);
				facetFIAreaLabel->SetEnabled(true);
				facetFlowArea->SetEditable(true);
				facetFILabel->SetEnabled(true);
				facetFlow->SetEditable(true);
				if (flowE) {
					if (f0->userOutgassing.length() == 0)
						facetFlow->SetText(f0->sh.outgassing*10.00); //10: Pa*m3/sec -> mbar*l/s
					else facetFlow->SetText(f0->userOutgassing);
				}
				else facetFlow->SetText("...");
				if (flowAreaE) facetFlowArea->SetText(f0->sh.outgassing / f0Area*10.00); else facetFlowArea->SetText("...");
				if (f0->sh.desorbType == 3) {
					facetDesTypeN->SetEditable(true);
					if (desorbTypeNE) facetDesTypeN->SetText(f0->sh.desorbTypeN); else facetDesTypeN->SetText("...");
				}
				else {
					facetDesTypeN->SetText("");
					facetDesTypeN->SetEditable(false);
				}

			}
			else { //No desorption
				facetFILabel->SetEnabled(false);
				facetFlow->SetEditable(false);
				facetFIAreaLabel->SetEnabled(false);
				facetFlowArea->SetEditable(false);
				facetDesTypeN->SetText("");
				facetDesTypeN->SetEditable(false);
				facetFlow->SetText("");
				facetFlowArea->SetText("");
			}
		}
		else { //Mixed state
			facetDesType->SetSelectedValue("...");
			facetFILabel->SetEnabled(false);
			facetFlow->SetEditable(false);
			facetFIAreaLabel->SetEnabled(false);
			facetFlowArea->SetEditable(false);
			facetDesTypeN->SetText("");
			facetDesTypeN->SetEditable(false);
			facetFlow->SetText("");
			facetFlowArea->SetText("");
		}

		if (facetAdvParams) facetAdvParams->Refresh(selectedFacets); //Refresh advanced facet parameters panel
		if (updateSelection) {

			if (nbSel > 1000 || geom->GetNbFacet() > 50000) { //If it would take too much time to look up every selected facet in the list
				facetList->ReOrder();
				facetList->SetSelectedRows(selectedFacets, false);
			}
			else {
				facetList->SetSelectedRows(selectedFacets, true);
			}
			facetList->lastRowSel = -1;
		}

		//Enabled->Editable
		facetSticking->SetEditable(true);
		facetOpacity->SetEditable(true);
		facetTemperature->SetEditable(true);
		facetSideType->SetEditable(true);
		facetDesType->SetEditable(true);
		facetProfileCombo->SetEditable(true);
		facetApplyBtn->SetEnabled(false);
	}
	else {
		ClearFacetParams();
		if (facetAdvParams) facetAdvParams->Refresh(std::vector<size_t>()); //Clear
		if (updateSelection) facetList->ClearSelection();
	}

	if (facetDetails) facetDetails->Update();
	if (facetCoordinates) facetCoordinates->UpdateFromSelection();
	if (texturePlotter) texturePlotter->Update(m_fTime, true); //Facet change
	if (outgassingMapWindow) outgassingMapWindow->Update(m_fTime, true);
	if (histogramSettings) histogramSettings->Refresh(selectedFacets);
}

// Name: FrameMove()
// Desc: Called once per frame, the call is the entry point for animating
//       the scene.

int MolFlow::FrameMove()
{
    bool runningState = worker.IsRunning();
    double elapsedTime = worker.simuTimer.Elapsed();
	if (runningState && ((m_fTime - lastUpdate) >= 1.0f)) {
		if (textureScaling) textureScaling->Update();
		//if (formulaEditor && formulaEditor->IsVisible()) formulaEditor->Refresh(); //Interface::Framemove does it already
	}
	Interface::FrameMove(); //might reset lastupdate
	char tmp[256];
	if (globalSettings) globalSettings->SMPUpdate();

	if ((elapsedTime <= 2.0f) && runningState) {
		hitNumber->SetText("Starting...");
		desNumber->SetText("Starting...");
	}
	else {
	    double current_avg = hps.avg();
        if(!runningState)
            current_avg = hps_runtotal.avg();
        else
            current_avg = (current_avg != 0.0) ? current_avg : hps.last();

        sprintf(tmp, "%s (%s)", Util::formatInt(worker.globalHitCache.globalHits.nbMCHit, "hit"),
                Util::formatPs(current_avg, "hit"));
        hitNumber->SetText(tmp);

        current_avg = dps.avg();
        if(!runningState) current_avg = dps_runtotal.avg();
        else current_avg = (current_avg != 0.0) ? current_avg : dps.last();

        sprintf(tmp, "%s (%s)", Util::formatInt(worker.globalHitCache.globalHits.nbDesorbed, "des"),
                Util::formatPs(current_avg, "des"));
        desNumber->SetText(tmp);
	}

    //sprintf(tmp, "Running: %s", Util::formatTime(worker.simuTimer.Elapsed()));


	// Save previous state to react to changes
    prevRunningState = runningState;

    return GL_OK;
}

// Name: RestoreDeviceObjects()
// Desc: Initialize scene objects.

int MolFlow::RestoreDeviceObjects()
{
	RestoreDeviceObjects_shared();

	//Different Molflow implementations:
	RVALIDATE_DLG(facetAdvParams);
	RVALIDATE_DLG(facetDetails);
	RVALIDATE_DLG(smartSelection);
	RVALIDATE_DLG(viewer3DSettings);
	RVALIDATE_DLG(textureScaling);
	RVALIDATE_DLG(globalSettings);
	RVALIDATE_DLG(profilePlotter);
	RVALIDATE_DLG(texturePlotter);

	//Molflow only:
	RVALIDATE_DLG(importDesorption);
	RVALIDATE_DLG(timeSettings);
	RVALIDATE_DLG(movement);
	RVALIDATE_DLG(outgassingMapWindow);
	RVALIDATE_DLG(parameterEditor);
	RVALIDATE_DLG(pressureEvolution);
	RVALIDATE_DLG(timewisePlotter);

	return GL_OK;
}

// Name: InvalidateDeviceObjects()
// Desc: Free all alocated resource

int MolFlow::InvalidateDeviceObjects()
{
	InvalidateDeviceObjects_shared();

	//Different Molflow implementations:
	IVALIDATE_DLG(facetAdvParams);
	IVALIDATE_DLG(facetDetails);
	IVALIDATE_DLG(smartSelection);
	IVALIDATE_DLG(viewer3DSettings);
	IVALIDATE_DLG(textureScaling);
	IVALIDATE_DLG(globalSettings);
	IVALIDATE_DLG(profilePlotter);
	IVALIDATE_DLG(texturePlotter);

	//Molflow only:
	IVALIDATE_DLG(importDesorption);
	IVALIDATE_DLG(timeSettings);
	IVALIDATE_DLG(movement);
	IVALIDATE_DLG(outgassingMapWindow);
	IVALIDATE_DLG(parameterEditor);
	IVALIDATE_DLG(pressureEvolution);
	IVALIDATE_DLG(timewisePlotter);

	return GL_OK;
}

void MolFlow::ExportProfiles() {

	Geometry *geom = worker.GetGeometry();
	if (geom->GetNbSelectedFacets() == 0) {
		GLMessageBox::Display("Empty selection", "Error", GLDLG_OK, GLDLG_ICONERROR);
		return;
	}

	//FILENAME *fn = GLFileBox::SaveFile(currentDir, nullptr, "Save File", fileProfFilters, 0);
	std::string saveFile = NFD_SaveFile_Cpp(fileProfFilters, "");

	if (!saveFile.empty()) {
		try {
			worker.ExportProfiles(saveFile.c_str());
		}
		catch(const std::exception &e) {
			char errMsg[512];
			sprintf(errMsg, "%s\nFile:%s", e.what(), saveFile.c_str());
			GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
		}
	}
}

void MolFlow::ExportAngleMaps() {

	std::string profFile = NFD_SaveFile_Cpp(fileProfFilters, "");

	if (!profFile.empty()) {
		try {
			auto retVal = worker.ExportAngleMaps(profFile, false);
			if (!retVal) return; //user cancel or error
			std::vector<std::string> exportList = *retVal; //vector of fileNames, might be empty
            if (exportList.empty()) {
                GLMessageBox::Display("Select at least one facet with recorded angle map", "Error", GLDLG_OK, GLDLG_ICONERROR);
                return;
            }
		}
		catch(const std::exception &e) {
			char errMsg[512];
			sprintf(errMsg, "%s\nFile:%s", e.what(), profFile.c_str());
			GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
	}
}

void MolFlow::ImportAngleMaps(){
	std::vector<size_t> selFacets = worker.GetGeometry()->GetSelectedFacets();
	if (selFacets.empty()) {
		GLMessageBox::Display("Select at least one facet to import angle map to", "Error", GLDLG_OK, GLDLG_ICONERROR);
		return;
	}

	//std::vector<FILENAME> files = GLFileBox::OpenMultipleFiles("CSV files\0*.csv\0All files\0*.*\0", "Import angle map(s)");
	auto fileNames = NFD_OpenMultiple_Cpp("csv", "");
	if (fileNames.empty()) return;
	if (selFacets.size() != fileNames.size()) {
		GLMessageBox::Display("Select the same number of facets and files to import", "Error", GLDLG_OK, GLDLG_ICONERROR);
		return;
	}
	for (size_t i = 0; i < fileNames.size();i++) {
		try {
			auto *f = new FileReader(fileNames[i]);
			std::vector<std::vector<std::string>> table = f->ImportCSV_string();
			SAFE_DELETE(f);
			AskToReset(&worker);
			worker.GetGeometry()->GetFacet(selFacets[i])->ImportAngleMap(table);
			worker.Reload();
		}
		catch(const std::exception &e) {
				char errMsg[512];
				sprintf(errMsg, "%s\nFile:%s", e.what(), fileNames[i].c_str());
				GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
				return;
		}
	}
}

void MolFlow::CopyAngleMapToClipboard()
{
	Geometry *geom = worker.GetGeometry();
	size_t angleMapFacetIndex;
	bool found = false;
	for (size_t i = 0; i < geom->GetNbFacet(); i++) {
		InterfaceFacet* f = geom->GetFacet(i);
		if (f->selected && !f->angleMapCache.empty()) {
			if (found) {
				GLMessageBox::Display("More than one facet with recorded angle map selected", "Error", GLDLG_OK, GLDLG_ICONERROR);
				return;
			}
			angleMapFacetIndex = i;
			found = true;
		}
	}
	if (!found) {
		GLMessageBox::Display("No facet with recorded angle map selected", "Error", GLDLG_OK, GLDLG_ICONERROR);
		return;
	}

	/*
	if (!worker.IsDpInitialized()) {
		GLMessageBox::Display("Worker Dataport not initialized yet", "Error", GLDLG_OK, GLDLG_ICONERROR);
		return;
	}*/

		try {
			std::string map = geom->GetFacet(angleMapFacetIndex)->GetAngleMap(2); //Clipboard format: tab-separated (format==2)
			if (map.length() > (10 * 1024 * 1024)) {
				if (GLMessageBox::Display("Angle map text over 10MB. Copy to clipboard?", "Warning", GLDLG_OK | GLDLG_CANCEL, GLDLG_ICONWARNING) != GLDLG_OK)
					return;
			}

			GLToolkit::CopyTextToClipboard(map);

		}
		catch(const std::exception &e) {
			GLMessageBox::Display(e.what(), "Error", GLDLG_OK, GLDLG_ICONERROR);
		}
}

void MolFlow::ClearAngleMapsOnSelection() {
	//if (AskToReset()) {
		Geometry *geom = worker.GetGeometry();
		for (size_t i = 0; i < geom->GetNbFacet(); i++) {
			InterfaceFacet* f = geom->GetFacet(i);
			if (f->selected && !f->angleMapCache.empty()) {
				f->angleMapCache.clear();
			}
		}
	//}
}

/*
void MolFlow::ImportDesorption_DES() {

	FILENAME *fn = GLFileBox::OpenFile(currentDir, nullptr, "Import desorption File", fileDesFilters, 0);

	if (fn) {

		try {
			worker.ImportDesorption_DES(fn->fullName);
		}
		catch(const std::exception &e) {
			char errMsg[512];
			sprintf(errMsg, "%s\nFile:%s", e.what(), fn->fullName);
			GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
		}
		worker.CalcTotalOutgassing();
		UpdateFacetParams();
	}

}
*/

void MolFlow::SaveFile() {
	if (!worker.fullFileName.empty()) {

		auto *progressDlg2 = new GLProgress("Saving...", "Please wait");
		progressDlg2->SetProgress(0.5);
		progressDlg2->SetVisible(true);
		//GLWindowManager::Repaint();

		try {
			worker.SaveGeometry(worker.fullFileName, progressDlg2, false);
			ResetAutoSaveTimer();
		}
		catch(const std::exception &e) {
			char errMsg[512];
			sprintf(errMsg, "%s\nFile:%s", e.what(), worker.GetCurrentFileName().c_str());
			GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
		}
		progressDlg2->SetVisible(false);
		SAFE_DELETE(progressDlg2);
		changedSinceSave = false;

	}
	else SaveFileAs();
}

void MolFlow::LoadFile(const std::string &fileName) {

	std::string fileShortName, filePath;

	if (fileName.empty()) {
        filePath = NFD_OpenFile_Cpp(fileLoadFilters, "");
	}
	else{
        filePath = fileName;
	}

	if (filePath.empty()) return; //User closed Open... dialog

	if (!FileUtils::Exist(filePath)) {
		auto answer = GLMessageBox::Display(
			fmt::format("{}\nDoesn't exist. Remove from the Recent files menu?",filePath),
			"No such file",
			{ "Yes","No" },
			GLDLG_ICONERROR);
		if (answer == 0) { //"Yes"
			RemoveRecent(filePath.c_str());
		}
		return;
	}

	auto *progressDlg2 = new GLProgress("Preparing to load file...", "Please wait");
	progressDlg2->SetVisible(true);
	progressDlg2->SetProgress(0.0);
	//GLWindowManager::Repaint();

	fileShortName = FileUtils::GetFilename(filePath);

	try {
		ClearFormulas();
		ClearParameters();
		ClearAllSelections();
		ClearAllViews();
		ResetSimulation(false);

		worker.LoadGeometry(filePath);

		Geometry *geom = worker.GetGeometry();

		
		// Default initialisation
		for (auto & view : viewer)
			view->SetWorker(&worker);

		//UpdateModelParams();
		startSimu->SetEnabled(true);
		//compACBtn->SetEnabled(modeCombo->GetSelectedIndex() == 1);
		//singleACBtn->SetEnabled(modeCombo->GetSelectedIndex() == 1);
		//resetSimu->SetEnabled(true);
		ClearFacetParams();
		nbDesStart = worker.globalHitCache.globalHits.nbDesorbed;
		nbHitStart = worker.globalHitCache.globalHits.nbMCHit;
		AddRecent(filePath.c_str());
		geom->viewStruct = -1;

		UpdateStructMenu();
		UpdateCurrentDir(filePath.c_str());

		// Check non simple polygon
		progressDlg2->SetMessage("Checking for non simple polygons...");

		geom->CheckCollinear();
		//geom->CheckNonSimple();
		geom->CheckIsolatedVertex();
		// Set up view
		// Default
		viewer[0]->SetProjection(ORTHOGRAPHIC_PROJ);
		viewer[0]->ToFrontView();
		viewer[1]->SetProjection(ORTHOGRAPHIC_PROJ);
		viewer[1]->ToTopView();
		viewer[2]->SetProjection(ORTHOGRAPHIC_PROJ);
		viewer[2]->ToSideView();
		viewer[3]->SetProjection(PERSPECTIVE_PROJ);
		viewer[3]->ToFrontView();
		SelectViewer(0);

		ResetAutoSaveTimer();
		//UpdatePlotters();

		if (timeSettings) timeSettings->RefreshMoments();
		if (momentsEditor) momentsEditor->Refresh();
		if (pressureEvolution) pressureEvolution->Reset();
		if (timewisePlotter) timewisePlotter->Refresh();
		if (histogramPlotter) histogramPlotter->Reset();
		if (profilePlotter) profilePlotter->Refresh();
        if (convergencePlotter) convergencePlotter->Refresh();
        if (texturePlotter) texturePlotter->Update(0.0,true);
		//if (parameterEditor) parameterEditor->UpdateCombo(); //Done by ClearParameters()
		if (textureScaling) textureScaling->Update();
		if (outgassingMapWindow) outgassingMapWindow->Update(m_fTime, true);
		if (facetDetails) facetDetails->Update();
		if (facetCoordinates) facetCoordinates->UpdateFromSelection();
		if (vertexCoordinates) vertexCoordinates->Update();
		if (movement) movement->Update();
		if (globalSettings && globalSettings->IsVisible()) globalSettings->Update();
		if (formulaEditor) formulaEditor->Refresh();
		if (parameterEditor) parameterEditor->Refresh();
	}
	catch(const std::exception &e) {

		char errMsg[512];
		sprintf(errMsg, "%s\nFile:%s", e.what(), fileShortName.c_str());
		GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
		RemoveRecent(filePath.c_str());

	}
	progressDlg2->SetVisible(false);
	SAFE_DELETE(progressDlg2);
	changedSinceSave = false;
}

void MolFlow::InsertGeometry(bool newStr, const std::string &fileName) {
	

	std::string fileShortName, filePath;

	if (fileName.empty()) {
        filePath = NFD_OpenFile_Cpp(fileInsertFilters, "");
	}
	else{
        filePath = fileName;
    }


	auto *progressDlg2 = new GLProgress("Preparing to load file...", "Please wait");
	progressDlg2->SetVisible(true);
	progressDlg2->SetProgress(0.0);
	//GLWindowManager::Repaint();

	if (filePath.empty()) {
		progressDlg2->SetVisible(false);
		SAFE_DELETE(progressDlg2);
		return;
	}
	
	if (!AskToReset()) return;
	ResetSimulation(false);

	fileShortName = FileUtils::GetFilename(filePath);

	try {

		//worker.InsertGeometry(newStr, fullName);
		worker.LoadGeometry(filePath, true, newStr);

		Geometry *geom = worker.GetGeometry();
		worker.PrepareToRun();
		worker.CalcTotalOutgassing();

		//Increase BB
		for (auto & view : viewer)
			view->SetWorker(&worker);
		
		//UpdateModelParams();
		startSimu->SetEnabled(true);

		//compACBtn->SetEnabled(modeCombo->GetSelectedIndex() == 1);
		//singleACBtn->SetEnabled(modeCombo->GetSelectedIndex() == 1);
		//resetSimu->SetEnabled(true);
		//ClearFacetParams();
		//nbDesStart = worker.globState.globalHits.globalHits.hit.nbDesorbed;
		//nbHitStart = worker.globState.globalHits.globalHits.hit.nbMC;
		AddRecent(filePath.c_str());
		geom->viewStruct = -1;

		//worker.LoadTexturesGEO(fullName);
		UpdateStructMenu();
		
		//UpdateCurrentDir(fullName);

		geom->CheckCollinear();
		//geom->CheckNonSimple();
		geom->CheckIsolatedVertex();

		/*
		// Set up view
		// Default
		viewer[0]->SetProjection(ORTHOGRAPHIC_PROJ);
		viewer[0]->ToFrontView();
		viewer[1]->SetProjection(ORTHOGRAPHIC_PROJ);
		viewer[1]->ToTopView();
		viewer[2]->SetProjection(ORTHOGRAPHIC_PROJ);
		viewer[2]->ToSideView();
		viewer[3]->SetProjection(PERSPECTIVE_PROJ);
		viewer[3]->ToFrontView();
		SelectViewer(0);
		*/
		RefreshPlotterCombos();
		//UpdatePlotters();
		
		if (outgassingMapWindow) outgassingMapWindow->Update(m_fTime, true);
		if (facetDetails) facetDetails->Update();
		if (facetCoordinates) facetCoordinates->UpdateFromSelection();
		if (vertexCoordinates) vertexCoordinates->Update();
		if (formulaEditor) formulaEditor->Refresh();
		if (parameterEditor) parameterEditor->Refresh();
	}
	catch(const std::exception &e) {

		char errMsg[512];
		sprintf(errMsg, "%s\nFile:%s", e.what(), fileShortName.c_str());
		GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
		RemoveRecent(filePath.c_str());

	}
	progressDlg2->SetVisible(false);
	SAFE_DELETE(progressDlg2);
	changedSinceSave = true;
}

void MolFlow::ClearParameters() {
	auto iter = worker.parameters.begin();
	while (iter != worker.parameters.end()) {
		//Delete non-catalog parameters
		if (!iter->fromCatalog)
			iter = worker.parameters.erase(iter);
		else
			++iter;
	}
	if (parameterEditor) parameterEditor->Refresh();
}

void MolFlow::StartStopSimulation() {
	
	if (worker.globalHitCache.globalHits.nbMCHit <= 0 && !worker.model->wp.calcConstantFlow && worker.moments.empty()) {
		bool ok = GLMessageBox::Display("Warning: in the Moments Editor, the option \"Calculate constant flow\" is disabled.\n"
			"This is useful for time-dependent simulations.\n"
			"However, you didn't define any moments, suggesting you're using steady-state mode.\n"
			"\nDo you want to continue?\n", "Strange time settings", GLDLG_OK | GLDLG_CANCEL, GLDLG_ICONWARNING) == GLDLG_OK;
		if (!ok) return;
	}

    worker.StartStop(m_fTime);
	if (!worker.IsRunning()) { //Force update on simulation stop
        formula_ptr->UpdateFormulaValues(worker.globalHitCache.globalHits.nbDesorbed);
        UpdatePlotters();
		//if (autoUpdateFormulas) UpdateFormula();
		if (autoUpdateFormulas && formulaEditor && formulaEditor->IsVisible()) formulaEditor->UpdateValues();
		if (particleLogger && particleLogger->IsVisible()) particleLogger->UpdateStatus();
	}

	// Frame rate measurement
	// reset on start only
	if(worker.IsRunning()) {
	    hps.clear();
        dps.clear();
        hps_runtotal.clear();
        dps_runtotal.clear();
    }

    lastMeasTime = m_Timer.Elapsed();
    lastNbHit = worker.globalHitCache.globalHits.nbMCHit;
	lastNbDes = worker.globalHitCache.globalHits.nbDesorbed;

	hps_runtotal.push(lastNbHit, lastMeasTime);
	dps_runtotal.push(lastNbDes, lastMeasTime);
	lastUpdate = 0.0;
}

// Name: EventProc()
// Desc: Message proc function to handle key and mouse input

void MolFlow::ProcessMessage(GLComponent *src, int message)
{

	if (ProcessMessage_shared(src, message)) return; //Already processed by common interface

	Geometry *geom = worker.GetGeometry();
	switch (message) {

		//MENU --------------------------------------------------------------------
	case MSG_MENU:
		switch (src->GetId()) {

		case MENU_FILE_IMPORTDES_SYN:

			if (geom->IsLoaded()) {
				if (!importDesorption) importDesorption = new ImportDesorption();
				importDesorption->SetGeometry(geom, &worker);
				importDesorption->SetVisible(true);
			}
			else GLMessageBox::Display("No geometry loaded.", "No geometry", GLDLG_OK, GLDLG_ICONERROR);
			break;
		/* //Deprecated
		case MENU_FILE_IMPORTDES_DES:
			ImportDesorption_DES();
			break;
		*/

		case MENU_FILE_EXPORTTEXTURE_AREA:
			ExportTextures(0, 0); break;
		case MENU_FILE_EXPORTTEXTURE_MCHITS:
			ExportTextures(0, 1); break;
		case MENU_FILE_EXPORTTEXTURE_IMPINGEMENT:
			ExportTextures(0, 2); break;
		case MENU_FILE_EXPORTTEXTURE_PART_DENSITY:
			ExportTextures(0, 3); break;
		case MENU_FILE_EXPORTTEXTURE_GAS_DENSITY:
			ExportTextures(0, 4); break;
		case MENU_FILE_EXPORTTEXTURE_PRESSURE:
			ExportTextures(0, 5); break;
		case MENU_FILE_EXPORTTEXTURE_AVG_V:
			ExportTextures(0, 6); break;
		case MENU_FILE_EXPORTTEXTURE_V_VECTOR:
			ExportTextures(0, 7); break;
		case MENU_FILE_EXPORTTEXTURE_N_VECTORS:
			ExportTextures(0, 8); break;
		case MENU_FILE_EXPORTTEXTURE_AREA_COORD:
			ExportTextures(1, 0); break;
		case MENU_FILE_EXPORTTEXTURE_MCHITS_COORD:
			ExportTextures(1, 1); break;
		case MENU_FILE_EXPORTTEXTURE_IMPINGEMENT_COORD:
			ExportTextures(1, 2); break;
		case MENU_FILE_EXPORTTEXTURE_PART_DENSITY_COORD:
			ExportTextures(1, 3); break;
		case MENU_FILE_EXPORTTEXTURE_GAS_DENSITY_COORD:
			ExportTextures(1, 4); break;
		case MENU_FILE_EXPORTTEXTURE_PRESSURE_COORD:
			ExportTextures(1, 5); break;
		case MENU_FILE_EXPORTTEXTURE_AVG_V_COORD:
			ExportTextures(1, 6); break;
		case MENU_FILE_EXPORTTEXTURE_V_VECTOR_COORD:
			ExportTextures(1, 7); break;
		case MENU_FILE_EXPORTTEXTURE_N_VECTORS_COORD:
			ExportTextures(1, 8); break;

		case MENU_FILE_EXPORTPROFILES:
			ExportProfiles();
			break;

		case MENU_TOOLS_MOVINGPARTS:
			if (!movement) movement = new Movement(geom, &worker);
			movement->Update();
			movement->SetVisible(true);
			break;

		case MENU_EDIT_TSCALING:
			if (!textureScaling || !textureScaling->IsVisible()) {
				SAFE_DELETE(textureScaling);
				textureScaling = new TextureScaling();
				textureScaling->Display(&worker, viewer);
			}
			break;

		case MENU_EDIT_GLOBALSETTINGS:
		    if(imWnd) imWnd->ToggleGlobalSettings();
			/*if (!globalSettings) globalSettings = new GlobalSettings(&worker);
			globalSettings->Update();
			globalSettings->SetVisible(true);*/
			break;

		case MENU_TOOLS_PROFPLOTTER:
			if (!profilePlotter) profilePlotter = new ProfilePlotter();
			profilePlotter->Display(&worker);
			break;
        case MENU_TOOLS_TEXPLOTTER:
			if (!texturePlotter) texturePlotter = new TexturePlotter();
			texturePlotter->Display(&worker);
			break;
		case MENU_FACET_DETAILS:
			if (facetDetails == nullptr) facetDetails = new FacetDetails();
			facetDetails->Display(&worker);
			break;

		case MENU_TIME_PRESSUREEVOLUTION:
			if (!pressureEvolution) pressureEvolution = new PressureEvolution(&worker);
			pressureEvolution->SetVisible(true);
			break;

		case MENU_FACET_OUTGASSINGMAP:
			if (!outgassingMapWindow) outgassingMapWindow = new OutgassingMapWindow();
			outgassingMapWindow->Display(&worker);
			break;
		case MENU_FACET_REMOVESEL:
		{
			auto selectedFacets = geom->GetSelectedFacets();
			if (selectedFacets.empty()) return; //Nothing selected
			if (GLMessageBox::Display("Remove selected facets?", "Question", GLDLG_OK | GLDLG_CANCEL, GLDLG_ICONINFO) == GLDLG_OK) {
				if (AskToReset()) {
					if (worker.IsRunning()) worker.Stop_Public();
					geom->RemoveFacets(selectedFacets);
					worker.CalcTotalOutgassing();
					//geom->CheckIsolatedVertex();
					UpdateModelParams();
					RefreshPlotterCombos();
					//UpdatePlotters();
					if (vertexCoordinates) vertexCoordinates->Update();
					if (facetCoordinates) facetCoordinates->UpdateFromSelection();
					// Send to sub process
					worker.Reload();
				}
			}
			break;
		}
		case MENU_FACET_SELECTSTICK:
			geom->UnselectAll();
			for (int i = 0; i < geom->GetNbFacet(); i++)
				if (geom->GetFacet(i)->sh.sticking_paramId!=-1 || (geom->GetFacet(i)->sh.sticking != 0.0 && !geom->GetFacet(i)->IsTXTLinkFacet()))
					geom->SelectFacet(i);
			geom->UpdateSelection();
			UpdateFacetParams(true);
			break;

		case MENU_FACET_SELECTREFL:
			geom->UnselectAll();
			for (int i = 0; i < geom->GetNbFacet(); i++) {
				InterfaceFacet *f = geom->GetFacet(i);
				if (f->sh.desorbType == DES_NONE && f->sh.sticking == 0.0 && f->sh.opacity > 0.0)
					geom->SelectFacet(i);
			}
			geom->UpdateSelection();
			UpdateFacetParams(true);
			break;

		case MENU_FACET_SELECTVOL:
			geom->UnselectAll();
			for (int i = 0; i < geom->GetNbFacet(); i++)
				if (geom->GetFacet(i)->sh.isVolatile)
					geom->SelectFacet(i);
			geom->UpdateSelection();
			UpdateFacetParams(true);
			break;

		case MENU_FACET_SELECTDES:
			geom->UnselectAll();
			for (int i = 0; i < geom->GetNbFacet(); i++)
				if (geom->GetFacet(i)->sh.desorbType != DES_NONE)
					geom->SelectFacet(i);
			geom->UpdateSelection();
			UpdateFacetParams(true);
			break;
		case MENU_SELECT_HASDESFILE:
			geom->UnselectAll();
			for (int i = 0; i < geom->GetNbFacet(); i++)
				if (geom->GetFacet(i)->hasOutgassingFile)
					geom->SelectFacet(i);
			geom->UpdateSelection();
			UpdateFacetParams(true);
			break;

		case MENU_TIME_SETTINGS:
			if (!timeSettings) timeSettings = new TimeSettings(&worker);
			timeSettings->SetVisible(true);
			break;
		case MENU_TIME_MOMENTS_EDITOR:
			if (!momentsEditor || !momentsEditor->IsVisible()) {
				SAFE_DELETE(momentsEditor);
				momentsEditor = new MomentsEditor(&worker);
				momentsEditor->Refresh();
				momentsEditor->SetVisible(true);
			}
			break;
		case MENU_TIME_PARAMETER_EDITOR:
			if (!parameterEditor) parameterEditor = new ParameterEditor(&worker);
			parameterEditor->SetVisible(true);
			break;
		case MENU_TIMEWISE_PLOTTER:
			if (!timewisePlotter) timewisePlotter = new TimewisePlotter();
			timewisePlotter->Display(&worker);
			break;
		case MENU_VERTEX_REMOVE:
			if (geom->IsLoaded()) {
				if (GLMessageBox::Display("Remove Selected vertices?\nNote: It will also affect facets that contain them!", "Question", GLDLG_OK | GLDLG_CANCEL, GLDLG_ICONINFO) == GLDLG_OK) {
					if (AskToReset()) {
						if (worker.IsRunning()) worker.Stop_Public();
						geom->RemoveSelectedVertex();
						worker.CalcTotalOutgassing();
						geom->Rebuild(); //Will recalculate facet parameters
						UpdateModelParams();
						if (vertexCoordinates) vertexCoordinates->Update();
						if (facetCoordinates) facetCoordinates->UpdateFromSelection();
						if (profilePlotter) profilePlotter->Refresh();
						if (pressureEvolution) pressureEvolution->Refresh();
						if (timewisePlotter) timewisePlotter->Refresh();
						if (facetCoordinates) facetCoordinates->UpdateFromSelection();
						if (vertexCoordinates) vertexCoordinates->Update();
						// Send to sub process
						worker.Reload();
					}
				}

			}
			else GLMessageBox::Display("No geometry loaded.", "No geometry", GLDLG_OK, GLDLG_ICONERROR);
			break;
		}

		//TEXT --------------------------------------------------------------------
	case MSG_TEXT_UPD:
		if (src == facetSticking || src == facetTemperature) {
			calcFlow();
			facetApplyBtn->SetEnabled(true);
		}
		else if (src == facetOpacity || src == facetDesTypeN) {
			facetApplyBtn->SetEnabled(true);
		}
		else if (src == facetFlow) {
			double outgassing;
			double area;
			facetFlow->GetNumber(&outgassing);
			facetArea->GetNumber(&area);
			if (area == 0) facetFlowArea->SetText("#DIV0");
			else facetFlowArea->SetText(outgassing / area);
			facetApplyBtn->SetEnabled(true);
			facetFILabel->SetState(true);
			facetFIAreaLabel->SetState(false);
			// //else if ( src == facetMass ) {
			//  facetApplyBtn->SetEnabled(true);
		}
		else if (src == facetFlowArea) {
			double flowPerArea;
			double area;
			facetFlowArea->GetNumber(&flowPerArea);
			facetArea->GetNumber(&area);
			facetFlow->SetText(flowPerArea*area);
			facetApplyBtn->SetEnabled(true);
			facetFIAreaLabel->SetState(true);
			facetFILabel->SetState(false);
			// //else if ( src == facetMass ) {
			//  facetApplyBtn->SetEnabled(true);
		}
		else if (src == facetPumping) {
			calcSticking();
			facetApplyBtn->SetEnabled(true);
			// //else if ( src == facetMass ) {
			//  facetApplyBtn->SetEnabled(true);
		}
		break;

	case MSG_TEXT:
		if (src == facetSticking || src == facetDesTypeN || src == facetOpacity || src == facetTemperature
		|| src == facetPumping || src == facetFlow || src == facetFlowArea) {
			ApplyFacetParams();
		}
		break;

		//COMBO -------------------------------------------------------------------
	case MSG_COMBO:

		if (src == facetDesType) {
			facetApplyBtn->SetEnabled(true);
			bool hasDesorption = facetDesType->GetSelectedIndex() != 0;

			facetFlow->SetEditable(hasDesorption);
			facetFlowArea->SetEditable(hasDesorption);

			int color = (hasDesorption) ? 0 : 110;
			facetFILabel->SetTextColor(color, color, color);
			facetFIAreaLabel->SetTextColor(color, color, color);
			facetFILabel->SetEnabled(hasDesorption);
			facetFIAreaLabel->SetEnabled(hasDesorption);
			facetDesTypeN->SetEditable(facetDesType->GetSelectedIndex() == 3);
		}
		else if (src == facetProfileCombo || src == facetSideType) {
			facetApplyBtn->SetEnabled(true);
		}
		break;

		//TOGGLE ------------------------------------------------------------------
	case MSG_TOGGLE:
		if (src == facetFILabel) {
			facetFILabel->SetState(true);
			facetFIAreaLabel->SetState(false);
		}
		else if (src == facetFIAreaLabel) {
			facetFILabel->SetState(false);
			facetFIAreaLabel->SetState(true);
		}
		else if (src == autoFrameMoveToggle) {
			autoFrameMove = autoFrameMoveToggle->GetState();
			forceFrameMoveButton->SetEnabled(!autoFrameMove);
		}
		else {
			// Update viewer flags
			UpdateViewerFlags();
		}
		break;

		//BUTTON ------------------------------------------------------------------
	case MSG_BUTTON:
		if (src == startSimu) {
			changedSinceSave = true;
			StartStopSimulation();
			resetSimu->SetEnabled(!worker.IsRunning());
		}

		else if (src == facetApplyBtn) {
			//changedSinceSave=true;
			ApplyFacetParams();
		}
		else if (src == facetDetailsBtn) {
			if (facetDetails == nullptr) facetDetails = new FacetDetails();
			facetDetails->Display(&worker);
		}
		else if (src == viewerMoreButton) {
			if (!viewer3DSettings)	viewer3DSettings = new Viewer3DSettings();
			viewer3DSettings->SetVisible(!viewer3DSettings->IsVisible());
			viewer3DSettings->Reposition();
			viewer3DSettings->Refresh(geom, viewer[curViewer]);

		}
		else if (src == textureScalingBtn) {
			if (!textureScaling || !textureScaling->IsVisible()) {
				SAFE_DELETE(textureScaling);
				textureScaling = new TextureScaling();
				textureScaling->Display(&worker, viewer);
			}
			else {
				textureScaling->SetVisible(false);
			}
		}
		else if (src == profilePlotterBtn) {
			if (!profilePlotter) profilePlotter = new ProfilePlotter();
			if (!profilePlotter->IsVisible()) profilePlotter->Display(&worker);
			else profilePlotter->SetVisible(false);
		}
		else if (src == texturePlotterBtn) {
			if (!texturePlotter) texturePlotter = new TexturePlotter();
			if (!texturePlotter->IsVisible()) texturePlotter->Display(&worker);
			else {
				texturePlotter->SetVisible(false);
				SAFE_DELETE(texturePlotter);
			}
		}
		else if (src == globalSettingsBtn) {
		    if(imWnd) imWnd->ToggleGlobalSettings();
			/*if (!globalSettings) globalSettings = new GlobalSettings(&worker);
			if (!globalSettings->IsVisible()) {
				globalSettings->Update();
				globalSettings->SetVisible(true);
			}
			else globalSettings->SetVisible(false);*/
		}
		else if (src == facetAdvParamsBtn) {
			if (!facetAdvParams) {
				facetAdvParams = new FacetAdvParams(&worker);
				facetAdvParams->Refresh(geom->GetSelectedFacets());
			}
			facetAdvParams->SetVisible(!facetAdvParams->IsVisible());
			facetAdvParams->Reposition();
		}
		/*else {
			ProcessFormulaButtons(src);
		}*/
		break;
	    default:
            //GLMessageBox::Display("Corrupted menu item selected!", "Error", GLDLG_OK, GLDLG_ICONERROR);
            return;
	}
}

void MolFlow::BuildPipe(double ratio, int steps) {

	char tmp[256];
	MolflowGeometry *geom = worker.GetMolflowGeometry();

	double R = 1.0;
	double L = ratio * R;
	int    step;
	if (steps) step = 5; //Quick Pipe
	else {
		sprintf(tmp, "100");
		//sprintf(title,"Pipe L/R = %g",L/R);
		char *nbFacets = GLInputBox::GetInput(tmp, "Number of facet", "Build Pipe");
		if (!nbFacets) return;
		if ((sscanf(nbFacets, "%d", &step) <= 0) || (step < 3)) {
			GLMessageBox::Display("Invalid number", "Error", GLDLG_OK, GLDLG_ICONERROR);
			return;
		}
	}
	std::ostringstream temp;
	temp << "PIPE" << L / R;
	geom->UpdateName(temp.str().c_str());

	try {
		geom->BuildPipe(L, R, 0, step);
        worker.needsReload = true;

		worker.CalcTotalOutgassing();
		//default values
		worker.model->wp.enableDecay = false;
		worker.model->wp.halfLife = 1;
		worker.model->wp.gasMass = 28;
		worker.ResetMoments();
		worker.model->wp.globalHistogramParams = HistogramParams();
        ResetSimulation(false);
        geom->InitializeGeometry();
    }
	catch(const std::exception &e) {
		GLMessageBox::Display((char *)e.what(), "Error building pipe", GLDLG_OK, GLDLG_ICONERROR);
		geom->Clear();
        ResetSimulation(false);
        return;
	}
	//worker.globState.globalHits.globalHits.hit.nbDesorbed = 0; //Already done by ResetWorkerStats
	//sprintf(tmp,"L|R %g",L/R);
	worker.SetCurrentFileName("");
	nbDesStart = 0;
	nbHitStart = 0;
	
	for (auto & view : viewer)
        view->SetWorker(&worker);
	
	//UpdateModelParams();
	startSimu->SetEnabled(true);
	//compACBtn->SetEnabled(modeCombo->GetSelectedIndex() == 1);
	//resetSimu->SetEnabled(true);
	ClearFacetParams();
	ClearFormulas();
	ClearParameters();
	ClearAllSelections();
	ClearAllViews();

	/*GLParser *f = new GLParser();
	f->SetExpression("A2/SUMDES");
	f->SetName("Trans. Prob.");
	f->Parse();*/
	AddFormula("Trans.prob.", "A2/SUMDES");

	UpdateStructMenu();
	// Send to sub process
	worker.Reload();

	//UpdatePlotters();

	if (timeSettings) timeSettings->RefreshMoments();
	if (momentsEditor) momentsEditor->Refresh();
	if (pressureEvolution) pressureEvolution->Reset();
	if (timewisePlotter) timewisePlotter->Refresh();
	if (profilePlotter) profilePlotter->Refresh();
	if (histogramSettings) histogramSettings->Refresh({});
	if (histogramPlotter) histogramPlotter->Reset();
	if (texturePlotter) texturePlotter->Update(0.0, true);
	//if (parameterEditor) parameterEditor->UpdateCombo(); //Done by ClearParameters()
	if (textureScaling) textureScaling->Update();
	if (outgassingMapWindow) outgassingMapWindow->Update(m_fTime, true);
	if (facetDetails) facetDetails->Update();
	if (facetCoordinates) facetCoordinates->UpdateFromSelection();
	if (vertexCoordinates) vertexCoordinates->Update();
	if (movement) movement->Update();
	if (globalSettings && globalSettings->IsVisible()) globalSettings->Update();
	if (formulaEditor) formulaEditor->Refresh();
	UpdateTitle();
	changedSinceSave = false;
	ResetAutoSaveTimer();
}

void MolFlow::EmptyGeometry() {

	Geometry *geom = worker.GetGeometry();
	ResetSimulation(false);

	try {
		geom->EmptyGeometry();
		worker.CalcTotalOutgassing();
		//default values
		worker.model->wp.enableDecay = false;
		worker.model->wp.halfLife = 1;
		worker.model->wp.gasMass = 28;
		worker.ResetMoments();
		worker.model->wp.globalHistogramParams = HistogramParams();
	}
	catch(const std::exception &e) {
		GLMessageBox::Display(e.what(), "Error resetting geometry", GLDLG_OK, GLDLG_ICONERROR);
		geom->Clear();
		return;
	}
	worker.SetCurrentFileName("");
	nbDesStart = 0;
	nbHitStart = 0;

	for (auto & view : viewer)
        view->SetWorker(&worker);

	//UpdateModelParams();
	startSimu->SetEnabled(true);
	//compACBtn->SetEnabled(modeCombo->GetSelectedIndex() == 1);
	//resetSimu->SetEnabled(true);
	ClearFacetParams();
	ClearFormulas();
	ClearParameters();
	ClearAllSelections();
	ClearAllViews();

	/*GLParser *f = new GLParser();
	f->SetExpression("A2/SUMDES");
	f->SetName("Trans. Prob.");
	f->Parse();*/
	//AddFormula("Trans.prob.", "A2/SUMDES");

	UpdateStructMenu();
	// Send to sub process
	worker.Reload();

	//UpdatePlotters();

	if (timeSettings) timeSettings->RefreshMoments();
	if (momentsEditor) momentsEditor->Refresh();
	if (pressureEvolution) pressureEvolution->Refresh();
	if (timewisePlotter) timewisePlotter->Refresh();
	if (profilePlotter) profilePlotter->Refresh();
	if (histogramSettings) histogramSettings->Refresh({});
	if (histogramPlotter) histogramPlotter->Reset();
	if (texturePlotter) texturePlotter->Update(0.0, true);
	//if (parameterEditor) parameterEditor->UpdateCombo(); //Done by ClearParameters()
	if (outgassingMapWindow) outgassingMapWindow->Update(m_fTime, true);
	if (movement) movement->Update();
	if (globalSettings && globalSettings->IsVisible()) globalSettings->Update();
	if (formulaEditor) formulaEditor->Refresh();
	
	if (textureScaling) textureScaling->Update();
	if (facetDetails) facetDetails->Update();
	if (facetCoordinates) facetCoordinates->UpdateFromSelection();
	if (vertexCoordinates) vertexCoordinates->Update();
	
	UpdateTitle();
	changedSinceSave = false;
	ResetAutoSaveTimer();
	RefreshPlotterCombos();
	//UpdatePlotters();
}

void MolFlow::LoadConfig() {

	FileReader *f = nullptr;
	char *w;

	try {

		f = new FileReader("molflow.cfg");
		MolflowGeometry *geom = worker.GetMolflowGeometry();

		f->ReadKeyword("showRules"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showRule = f->ReadInt();
		f->ReadKeyword("showNormals"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showNormal = f->ReadInt();
		f->ReadKeyword("showUV"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showUV = f->ReadInt();
		f->ReadKeyword("showLines"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showLine = f->ReadInt();
		f->ReadKeyword("showLeaks"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showLeak = f->ReadInt();
		f->ReadKeyword("showHits"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showHit = f->ReadInt();
		f->ReadKeyword("showVolume"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showVolume = f->ReadInt();
		f->ReadKeyword("showTexture"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showTexture = f->ReadInt();
        f->ReadKeyword("showFacetId"); f->ReadKeyword(":");
        for (auto & view : viewer)
            view->showFacetId = f->ReadInt();
		f->ReadKeyword("showFilter"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showFilter = f->ReadInt();
		f->ReadKeyword("showIndices"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showIndex = f->ReadInt();
		f->ReadKeyword("showVertices"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showVertexId = f->ReadInt();
		f->ReadKeyword("showMode"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showBack = f->ReadInt();
		f->ReadKeyword("showMesh"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showMesh = f->ReadInt();
		f->ReadKeyword("showHidden"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showHidden = f->ReadInt();
		f->ReadKeyword("showHiddenVertex"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showHiddenVertex = f->ReadInt();
		f->ReadKeyword("showTimeOverlay"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showTime = f->ReadInt();
		f->ReadKeyword("texColormap"); f->ReadKeyword(":");
		for (int i = 0; i < MAX_VIEWER; i++)
			//viewer[i]->showColormap = 
			f->ReadInt();
		f->ReadKeyword("translation"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->transStep = f->ReadDouble();
		f->ReadKeyword("dispNumLines"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->dispNumHits = f->ReadSizeT();
		f->ReadKeyword("dispNumLeaks"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->dispNumLeaks = f->ReadSizeT();
		f->ReadKeyword("dirShow"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->showDir = f->ReadInt();
		f->ReadKeyword("dirNorme"); f->ReadKeyword(":");
		geom->SetNormeRatio((float)f->ReadDouble());
		f->ReadKeyword("dirAutoNormalize"); f->ReadKeyword(":");
		geom->SetAutoNorme(f->ReadInt());
		f->ReadKeyword("dirCenter"); f->ReadKeyword(":");
		geom->SetCenterNorme(f->ReadInt());
		f->ReadKeyword("angle"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->angleStep = f->ReadDouble();
		f->ReadKeyword("autoScale"); f->ReadKeyword(":");
		geom->texAutoScale = f->ReadInt();
		f->ReadKeyword("autoScale_include_constant_flow"); f->ReadKeyword(":");
		geom->texAutoScaleIncludeConstantFlow = (short)f->ReadInt();

		f->ReadKeyword("textures_min_pressure_all"); f->ReadKeyword(":");
		geom->texture_limits[0].autoscale.min.steady_state = f->ReadDouble();
		f->ReadKeyword("textures_min_pressure_moments_only"); f->ReadKeyword(":");
		geom->texture_limits[0].autoscale.min.moments_only = f->ReadDouble();
		f->ReadKeyword("textures_max_pressure_all"); f->ReadKeyword(":");
		geom->texture_limits[0].autoscale.max.steady_state = f->ReadDouble();
		f->ReadKeyword("textures_max_pressure_moments_only"); f->ReadKeyword(":");
		geom->texture_limits[0].autoscale.max.moments_only = f->ReadDouble();

		f->ReadKeyword("textures_min_impingement_all"); f->ReadKeyword(":");
		geom->texture_limits[1].autoscale.min.steady_state = f->ReadDouble();
		f->ReadKeyword("textures_min_impingement_moments_only"); f->ReadKeyword(":");
		geom->texture_limits[1].autoscale.min.moments_only = f->ReadDouble();
		f->ReadKeyword("textures_max_impingement_all"); f->ReadKeyword(":");
		geom->texture_limits[1].autoscale.max.steady_state = f->ReadDouble();
		f->ReadKeyword("textures_max_impingement_moments_only"); f->ReadKeyword(":");
		geom->texture_limits[1].autoscale.max.moments_only = f->ReadDouble();

		f->ReadKeyword("textures_min_density_all"); f->ReadKeyword(":");
		geom->texture_limits[2].autoscale.min.steady_state = f->ReadDouble();
		f->ReadKeyword("textures_min_density_moments_only"); f->ReadKeyword(":");
		geom->texture_limits[2].autoscale.min.moments_only = f->ReadDouble();
		f->ReadKeyword("textures_max_density_all"); f->ReadKeyword(":");
		geom->texture_limits[2].autoscale.max.steady_state = f->ReadDouble();
		f->ReadKeyword("textures_max_density_moments_only"); f->ReadKeyword(":");
		geom->texture_limits[2].autoscale.max.moments_only = f->ReadDouble();

		f->ReadKeyword("processNum"); f->ReadKeyword(":");
		nbProc = f->ReadSizeT();
#if defined(_DEBUG)
		nbProc = 1;
#endif
		if (nbProc <= 0) nbProc = 1;
		f->ReadKeyword("recents"); f->ReadKeyword(":"); f->ReadKeyword("{");
		w = f->ReadString();
		while (strcmp(w, "}") != 0 && recentsList.size() < MAX_RECENT) {
			recentsList.emplace_back(strdup(w));
			w = f->ReadString();
		}

		f->ReadKeyword("cdir"); f->ReadKeyword(":");
		strcpy(currentDir, f->ReadString());
		f->ReadKeyword("cseldir"); f->ReadKeyword(":");
		strcpy(currentSelDir, f->ReadString());
		f->ReadKeyword("autonorme"); f->ReadKeyword(":");
		geom->SetAutoNorme(f->ReadInt());
		f->ReadKeyword("centernorme"); f->ReadKeyword(":");
		geom->SetCenterNorme(f->ReadInt());
		f->ReadKeyword("normeratio"); f->ReadKeyword(":");
		geom->SetNormeRatio((float)(f->ReadDouble()));
		f->ReadKeyword("autoSaveFrequency"); f->ReadKeyword(":");
		autoSaveFrequency = f->ReadDouble();
		f->ReadKeyword("autoSaveSimuOnly"); f->ReadKeyword(":");
		autoSaveSimuOnly = f->ReadInt();
		f->ReadKeyword("checkForUpdates"); f->ReadKeyword(":");
		/*checkForUpdates =*/ f->ReadInt(); //Old checkforupdates
		f->ReadKeyword("autoUpdateFormulas"); f->ReadKeyword(":");
		autoUpdateFormulas = f->ReadInt();
		f->ReadKeyword("compressSavedFiles"); f->ReadKeyword(":");
		compressSavedFiles = f->ReadInt();
		f->ReadKeyword("gasMass"); f->ReadKeyword(":");
		worker.model->wp.gasMass = f->ReadDouble();
		f->ReadKeyword("expandShortcutPanel"); f->ReadKeyword(":");
		bool isOpen = f->ReadInt();
		if (isOpen) shortcutPanel->Open();
		else shortcutPanel->Close();
		f->ReadKeyword("hideLot"); f->ReadKeyword(":");
		for (auto & view : viewer)
			view->hideLot = f->ReadInt();
		f->ReadKeyword("lowFluxMode"); f->ReadKeyword(":");
		worker.model->otfParams.lowFluxMode = f->ReadInt();
		f->ReadKeyword("lowFluxCutoff"); f->ReadKeyword(":");
		worker.model->otfParams.lowFluxCutoff = f->ReadDouble();
		f->ReadKeyword("leftHandedView"); f->ReadKeyword(":");
		leftHandedView = f->ReadInt();
		f->ReadKeyword("highlightNonplanarFacets"); f->ReadKeyword(":");
		highlightNonplanarFacets = f->ReadInt();
        f->ReadKeyword("highlightSelection"); f->ReadKeyword(":");
        highlightSelection = f->ReadInt();
        f->ReadKeyword("useOldXMLFormat"); f->ReadKeyword(":");
        useOldXMLFormat = f->ReadInt();
		f->ReadKeyword("showTP"); f->ReadKeyword(":");
		for (auto& view : viewer)
			view->showTP = f->ReadInt();
	}
	catch (...) {
		/*std::ostringstream tmp;
		tmp << err.GetMsg() << "\n\nThis is normal on the first launch and if you upgrade from an earlier version\n";
		tmp << "MolFlow will use default program settings.\nWhen you quit, a correct config file will be written\n";
		GLMessageBox::Display(tmp.str().c_str(), "Error loading config file", GLDLG_OK, GLDLG_ICONINFO);*/
		SAFE_DELETE(f);
		return;
	}
	SAFE_DELETE(f);
}


#define WRITEI(name,var) {             \
	f->Write(name);                      \
	f->Write(":");                       \
	for(size_t i=0;i<MAX_VIEWER;i++)        \
	f->Write(viewer[i]->var," ");   \
	f->Write("\n");                      \
}


#define WRITED(name,var) {             \
	f->Write(name);                      \
	f->Write(":");                       \
	for(size_t i=0;i<MAX_VIEWER;i++)        \
	f->Write(viewer[i]->var," ");\
	f->Write("\n");                      \
}


void MolFlow::SaveConfig() {

	FileWriter *f = nullptr;

	try {

		f = new FileWriter("molflow.cfg");
		MolflowGeometry *geom = worker.GetMolflowGeometry();

		// Save flags
		WRITEI("showRules", showRule);
		WRITEI("showNormals", showNormal);
		WRITEI("showUV", showUV);
		WRITEI("showLines", showLine);
		WRITEI("showLeaks", showLeak);
		WRITEI("showHits", showHit);
		WRITEI("showVolume", showVolume);
		WRITEI("showTexture", showTexture);
        WRITEI("showFacetId", showFacetId);
        WRITEI("showFilter", showFilter);
		WRITEI("showIndices", showIndex);
		WRITEI("showVertices", showVertexId);
		WRITEI("showMode", showBack);
		WRITEI("showMesh", showMesh);
		WRITEI("showHidden", showHidden);
		WRITEI("showHiddenVertex", showHiddenVertex);
		WRITEI("showTimeOverlay", showTime);
		//WRITEI("texColormap", showColormap);
		f->Write("texColormap:1 1 1 1\n");
		WRITED("translation", transStep);
		WRITEI("dispNumLines", dispNumHits);
		WRITEI("dispNumLeaks", dispNumLeaks);
		WRITEI("dirShow", showDir);
		f->Write("dirNorme:"); f->Write(geom->GetNormeRatio(), "\n");
		f->Write("dirAutoNormalize:"); f->Write(geom->GetAutoNorme(), "\n");
		f->Write("dirCenter:"); f->Write(geom->GetCenterNorme(), "\n");

		WRITED("angle", angleStep);
		f->Write("autoScale:"); f->Write(geom->texAutoScale, "\n");
		f->Write("autoScale_include_constant_flow:"); f->Write(geom->texAutoScaleIncludeConstantFlow, "\n");

		f->Write("textures_min_pressure_all:");
		f->Write(geom->texture_limits[0].autoscale.min.steady_state, "\n");
		f->Write("textures_min_pressure_moments_only:");
		f->Write(geom->texture_limits[0].autoscale.min.moments_only, "\n");
		f->Write("textures_max_pressure_all:");
		f->Write(geom->texture_limits[0].autoscale.max.steady_state, "\n");
		f->Write("textures_max_pressure_moments_only:");
		f->Write(geom->texture_limits[0].autoscale.max.moments_only, "\n");

		f->Write("textures_min_impingement_all:");
		f->Write(geom->texture_limits[1].autoscale.min.steady_state, "\n");

		f->Write("textures_min_impingement_moments_only:");
		f->Write(geom->texture_limits[1].autoscale.min.moments_only, "\n");
		f->Write("textures_max_impingement_all:");
		f->Write(geom->texture_limits[1].autoscale.max.steady_state, "\n");
		f->Write("textures_max_impingement_moments_only:");
		f->Write(geom->texture_limits[1].autoscale.max.moments_only, "\n");

		f->Write("textures_min_density_all:");
		f->Write(geom->texture_limits[2].autoscale.min.steady_state, "\n");
		f->Write("textures_min_density_moments_only:");
		f->Write(geom->texture_limits[2].autoscale.min.moments_only, "\n");
		f->Write("textures_max_density_all:");
		f->Write(geom->texture_limits[2].autoscale.max.steady_state, "\n");
		f->Write("textures_max_density_moments_only:");
		f->Write(geom->texture_limits[2].autoscale.max.moments_only, "\n");

#if defined(_DEBUG)
		f->Write("processNum:");f->Write(numCPU, "\n");
#else
		f->Write("processNum:"); f->Write(worker.GetProcNumber(), "\n");
#endif
		f->Write("recents:{\n");
		for(auto& recent : recentsList){
			f->Write("\"");
			f->Write(recent);
			f->Write("\"\n");
		}
		f->Write("}\n");

		f->Write("cdir:\""); f->Write(currentDir); f->Write("\"\n");
		f->Write("cseldir:\""); f->Write(currentSelDir); f->Write("\"\n");
		f->Write("autonorme:"); f->Write(geom->GetAutoNorme(), "\n");
		f->Write("centernorme:"); f->Write(geom->GetCenterNorme(), "\n");
		f->Write("normeratio:"); f->Write((double)(geom->GetNormeRatio()), "\n");
		
		f->Write("autoSaveFrequency:"); f->Write(autoSaveFrequency, "\n");
		f->Write("autoSaveSimuOnly:"); f->Write(autoSaveSimuOnly, "\n");
		f->Write("checkForUpdates:"); f->Write(/*checkForUpdates*/ 0, "\n"); //Deprecated
		f->Write("autoUpdateFormulas:"); f->Write(autoUpdateFormulas, "\n");
		f->Write("compressSavedFiles:"); f->Write(compressSavedFiles, "\n");
		f->Write("gasMass:"); f->Write(worker.model->wp.gasMass, "\n");
		f->Write("expandShortcutPanel:"); f->Write(!shortcutPanel->IsClosed(), "\n");

		WRITEI("hideLot", hideLot);
		f->Write("lowFluxMode:"); f->Write(worker.model->otfParams.lowFluxMode, "\n");
		f->Write("lowFluxCutoff:"); f->Write(worker.model->otfParams.lowFluxCutoff, "\n");
		f->Write("leftHandedView:"); f->Write(leftHandedView, "\n");
        f->Write("highlightNonplanarFacets:"); f->Write(highlightNonplanarFacets, "\n");
		f->Write("highlightSelection:"); f->Write(highlightSelection, "\n");
        f->Write("useOldXMLFormat:"); f->Write(useOldXMLFormat, "\n");
		WRITEI("showTP", showTP);
    }
	catch(const std::exception& e) {
		GLMessageBox::Display(e.what(), "Error saving config file", GLDLG_OK, GLDLG_ICONWARNING);
	}

	SAFE_DELETE(f);

}

void MolFlow::calcFlow() {
	double sticking;
	double area;
	double outgassing;
	double temperature;
	//double mass;

	facetSticking->GetNumber(&sticking);
	facetArea->GetNumber(&area);
	facetTemperature->GetNumber(&temperature);
	//facetMass->GetNumber(&mass);

	outgassing = 1 * sticking*area / 10.0 / 4.0*sqrt(8.0*8.31*temperature / PI / (worker.model->wp.gasMass*0.001));
	facetPumping->SetText(outgassing);
}

void MolFlow::calcSticking() {
	double sticking;
	double area;
	double outgassing;
	double temperature;
	//double mass;

	facetPumping->GetNumber(&outgassing);
	facetArea->GetNumber(&area);
	facetTemperature->GetNumber(&temperature);
	//facetMass->GetNumber(&mass);

	sticking = std::abs(outgassing / (area / 10.0)*4.0*sqrt(1.0 / 8.0 / 8.31 / (temperature)*PI*(worker.model->wp.gasMass*0.001)));
	facetSticking->SetText(sticking);
}

void MolFlow::CrashHandler(const std::exception &e) {
	char tmp[1024];
	sprintf(tmp, "Well, that's embarrassing. Molflow crashed and will exit now.\nBefore that, an autosave will be attempted.\nHere is the error info:\n\n%s", e.what());
    if(imWnd) {
        imWnd->renderSingle();
        imWnd->destruct();
        delete imWnd;
        imWnd = nullptr;
    }
    GLMessageBox::Display(tmp, "Main crash handler", GLDLG_OK, GLDGL_ICONDEAD);
	try {
		if (AutoSave(true))
			GLMessageBox::Display("Good news, autosave worked!", "Main crash handler", GLDLG_OK, GLDGL_ICONDEAD);
		else
			GLMessageBox::Display("Sorry, I couldn't even autosave.", "Main crash handler", GLDLG_OK, GLDGL_ICONDEAD);
	}
	catch(const std::exception& e) {
        sprintf(tmp, "Sorry, I couldn't even autosave:\n\n%s", e.what());
        GLMessageBox::Display(tmp, "Main crash handler", GLDLG_OK, GLDGL_ICONDEAD);
	}
}

void MolFlow::UpdatePlotters() {
    const bool forceUpdate = true; //worker.IsRunning();
	if (pressureEvolution) pressureEvolution->Update(m_fTime, forceUpdate);
	if (timewisePlotter) timewisePlotter->Update(m_fTime, forceUpdate);
	if (profilePlotter) profilePlotter->Update(m_fTime, forceUpdate);
	if (texturePlotter) texturePlotter->Update(m_fTime, forceUpdate);
	if (histogramPlotter) histogramPlotter->Update(m_fTime, forceUpdate);
	if (convergencePlotter) convergencePlotter->Update(m_fTime);
}

void MolFlow::RefreshPlotterCombos() {
	//Removes non-present views, rebuilds combobox and refreshes plotted data
	if (pressureEvolution) pressureEvolution->Refresh();
	if (timewisePlotter) timewisePlotter->Refresh();
	if (profilePlotter) profilePlotter->Refresh();
	if (histogramPlotter) histogramPlotter->Refresh();
    if (convergencePlotter) convergencePlotter->Refresh();
}

void MolFlow::UpdateFacetHits(bool allRows) {
	char tmp[256];
	Geometry *geom = worker.GetGeometry();

	try {
		// Facet list
		if (geom->IsLoaded()) {

			int sR, eR;
			if (allRows)
			{
				sR = 0;
				eR = (int)facetList->GetNbRow() - 1;
			}
			else
			{
				facetList->GetVisibleRows(&sR, &eR);
			}

			if (worker.displayedMoment == 0) {
				int colors[] = { COLOR_BLACK, COLOR_BLACK, COLOR_BLACK, COLOR_BLACK };
				facetList->SetColumnColors(colors);
			}
			else
			{
				int colors[] = { COLOR_BLACK, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE };
				facetList->SetColumnColors(colors);
			}

			for (int i = sR; i <= eR; i++) {
				int facetId = facetList->GetValueInt(i, 0) - 1;
				if (facetId == -2) facetId = (int)i;
				if (i >= geom->GetNbFacet()) {
					char errMsg[512];
					sprintf(errMsg, "Molflow::UpdateFacetHits()\nError while updating facet hits. Was looking for facet #%d (/%zu) in list.\nMolflow will now autosave and crash.", i + 1, geom->GetNbFacet());
					GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
					AutoSave();
                    throw std::runtime_error(errMsg);
				}
				InterfaceFacet *f = geom->GetFacet(facetId);
				sprintf(tmp, "%d", facetId + 1);
				facetList->SetValueAt(0, i, tmp);

                facetList->SetColumnLabel(1, "Hits");
                sprintf(tmp, "%zd", f->facetHitCache.nbMCHit);
                facetList->SetValueAt(1, i, tmp);
                sprintf(tmp, "%zd", f->facetHitCache.nbDesorbed);
                facetList->SetValueAt(2, i, tmp);
                sprintf(tmp, "%g", f->facetHitCache.nbAbsEquiv);
                facetList->SetValueAt(3, i, tmp);
			}

		}
	}
	catch(const std::exception &e) {
		char errMsg[512];
		sprintf(errMsg, "%s\nError while updating facet hits", e.what());
		GLMessageBox::Display(errMsg, "Error", GLDLG_OK, GLDLG_ICONERROR);
	}

}
