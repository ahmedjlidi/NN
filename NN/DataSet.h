#pragma once
#include "Tensor.h"

///////////////////////////////////////////////////////////
// Rx library
// Copyright (C) 2024 Ahmed Jlidi
// Contact: ahmed.jlidi543@gmail.com
//
//
// Anyone can use, modify, and distribute this software for any purpose.
// If you publish your app with this project, an acknowledgment of using it will be appreciated :D
//
// 
// If you change the original source code, you must clarify it and present it as altered version
// 
//
// This statement may not be removed or changed
//
// 
// Sources that helped me: 
// http://lodev.org/cgtutor/raycasting.html
// https://www.youtube.com/watch?v=NbSee-XM7WA&t=1445s
// 
// Copyright Thomas van der Berg, 2016
// 
///////////////////////////////////////////////////////////


//PreCompiled Header files must be used to run the library



namespace rx
{
	///ENUMS//////
	enum DESCRIBE
	{
		AVG,
		LABEL
	};


	//TYPE DEFINITIONS//////////////////////////////////////////////
	typedef std::pair<std::vector<std::string>, std::string> PAIR;
	typedef std::vector<PAIR> SET;
	typedef std::map<std::string, std::pair<int, int>> FreqTable;
	////////////////////////////////////////////////////////////////

	class DataSet
	{
	private:

		SET set;
		friend class Algorithm;
		std::unordered_map<std::string, std::string> cris;
		std::vector<std::string> labels;
		int rowNum = -1;

		//Frequency tables only work for binary datasets
		std::map<std::string, FreqTable>freq_tables;
		/////////////////////////////////////////////////

		//DESCRIBE FUNCTIONS////////////////////////////

		/*prints descritption about the average value from every column*/
		void describeAvg();

		/*prints descritption about the average value from a specific label*/
		void describeLabel(std::string _LABEL_);

		//Will be used by describeLabel
		std::string getAvgAtColumn(std::string _COLUMN_);
		/////////////////////////////////////////////////

		//UTILITY FUNCTIONS
		//////////////////////////////////////////
		template<typename T>
		bool findInVec(std::vector<T>, T);
		std::string cleanStr(std::string TEST);
		bool isInt(const std::string TEST);
		bool isFLoat(const std::string TEST);
		bool isStr(const std::string _TEST_);
		///////////////////////////////////////////////
	public:

		DataSet(std::vector<std::string>* cris = nullptr);

		/*load a csv file into the set*/
		bool loadCsvFile(const char filename[]);
		/*======================================*/

		/*removes any row with missing data*/
		void saniTize();
		/*===================================*/

		/*displays the first m rows of a dataset (dosen't work well with many labels) */
		void display(DataSet* dataset = nullptr, unsigned limit = 50);
		/*=================================================*/

		/*Prints a given row of dataset in formatted style*/
		void printPair(const PAIR& pair);
		/*============================================*/

		/*Gets the first pair it encouters with a key*/
		PAIR* getPairAt(std::string key);
		/*=========================================*/

		/*Removes a pair with a key*/
		void erasePairAt(std::string key);
		/*================================*/

		/*Clears whole dataset*/
		void clear();
		/*====================*/

		/*Returns pointer to the dataset*/
		SET* getSet();
		/*==============================*/

		PAIR* getPairAtIndex(int index);

		std::vector<std::string> valueAtIndex(int index);

		void setSet(rx::SET& newSet);

		/*Returns labels*/
		std::vector<std::string>& getLabels();
		/*=============================*/

		/*Gets output label*/
		std::string getOutputLabel();
		/*==========================*/

		/*Returns row numbers*/
		const int getRowNum() const;
		/*==========================*/

		//Frequqencey Table ( Works for binary datasets only)
		//////////////////////////////////////////////////////////////
		void initFreqTable(const std::string _ou1, const std::string _out2);
		FreqTable* getFreqTableAt(std::string key);
		void printFreqTableAt(std::string key);
		//////////////////////////////////////////////

		/*Returns a brief describing the output*/
		std::map<std::string, int> getOutputAnalisis();
		/*===========================================*/

		/*Describe the dataset*/
		void describe(short TYPE = AVG, std::string _LABEL_ = "");
		/*==========================================*/

		std::pair<Tensor, Tensor>get_As_Tensor();

		/*Returns number of rows and colums in the dataset*/
		void info();
		/*================================================*/


		void normalize(int x, int y);
	};
}

