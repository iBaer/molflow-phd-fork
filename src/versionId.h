#pragma once
#include <string>

#if defined(MOLFLOW)
//Hard-coded identifiers, update these on new release and rebuild solution
//---------------------------------------------------
static const std::string appName = "Molflow";
static const int appVersionId = 2906; //Compared with available updates. Global variable, so rebuild whole solution if changed.
static const std::string appVersionName = "2.9.6 (beta)";
//---------------------------------------------------
#if defined(_DEBUG)
static const std::string appTitle = "Molflow+ " + appVersionName + " debug version (" __DATE__ " " __TIME__ ")";
#else
static const std::string appTitle = "Molflow+ " + appVersionName + " (" __DATE__ ")";
#endif
#endif