#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include "caffe/util/base64.h"
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using std::vector;
using std::map;
using std::string;

namespace caffe {

// for fast readline
class TextFile
{
	vector<char> _buffer;

	int _bufPos;
	int _bytesInBuffer;

	FILE *_fp;

	void load_buffer();

public:
	TextFile();
	~TextFile();

	bool IsEOF();

	int Open(const char *fname, int buffer_size = 64 * 1024);
	void Close();

	void Seek(int64_t pos);
	bool ReadLine(string &line);
};

class TsvRawDataFile
{
	TextFile _dataFile;
	int _colData;
	int _colLabel;

	std::string _tsvFileName;
	std::vector<int64_t> _lineIndex;

	int _currentLine;

	// load line index file to vector lineIndex 
	int LoadLineIndex(const char *fileName, vector<int64_t> &lineIndex);

	string ChangeFileExtension(string filename, string ext);

public:
	~TsvRawDataFile();

	int Open(const char *fileName, int colData, int colLabel);
	void Close();
	void ShuffleData();

	bool IsEOF();
	void MoveToFirst();

	// read one line and do base64 decode
	int ReadNextDatum(vector<BYTE> &imbuf, int &label);
	// read lines to batch for parallel base64 decoding and image resizing.
	int ReadNextLine(vector<string> &base64codedImg, vector<int> &label);
};
}