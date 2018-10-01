#pragma once

#include <memory>
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
using std::shared_ptr;

namespace caffe {

string ChangeFileExtension(string filename, string ext);
  
// for fast readline
class TextFile
{
    vector<char> _buffer;
    bool _cacheAll;

    int64_t _bufPos;
    int64_t _bytesInBuffer;

    FILE *_fp;

    void load_buffer();

public:
	TextFile();
	~TextFile();

	bool IsEOF();

	int Open(const char *fname, bool cache_all);
	void Close();

	void Seek(int64_t pos);
	bool ReadLine(string &line);
};

void LoadLineIndex(const string &filename, vector<std::pair<int64_t, int64_t>>&lineIndex); 

class ITsvDataFile {
public:
    static ITsvDataFile* make_tsv(const char* fileName, bool cache_all, int colData, int colLabel);
    static ITsvDataFile* make_tsv(const vector<string> &fileNames, 
            const vector<bool> &cache_all,
            const vector<int> &colData,
            const vector<int> &colLabel);
    static ITsvDataFile* make_tsv(const vector<string> &fileNames, 
            bool cache_all,
            int colData,
            int colLabel);

	virtual void Close() = 0;
	virtual void ShuffleData(const string &filename) = 0;

	virtual bool IsEOF() = 0;
	virtual void MoveToFirst() = 0;
	virtual void MoveToLine(int lineNo) = 0;
    virtual void MoveToNext() = 0;
	virtual int TotalLines() = 0;

	// read lines to batch for parallel base64 decoding and image resizing.
	virtual int ReadNextLine(vector<string> &base64codedImg, vector<string> &label) = 0;
};

class TsvRawDataFile: public ITsvDataFile
{
	TextFile _dataFile;
	int _colData;
	int _colLabel;

	std::string _tsvFileName;
	std::vector<int64_t> _lineIndex;
    std::vector<int64_t> _shuffleLines;

	int _currentLine;

    // load line index file to vector lineIndex 
    int LoadLineIndex(const char *fileName, vector<int64_t> &lineIndex);

public:
	virtual ~TsvRawDataFile();

	int Open(const char *fileName, bool cache_all, int colData, int colLabel);
	void Close();
	void ShuffleData(const string &filename);

	bool IsEOF();
	void MoveToFirst();
	void MoveToLine(int lineNo);
    void MoveToNext();
	int TotalLines();

	// read lines to batch for parallel base64 decoding and image resizing.
	int ReadNextLine(vector<string> &base64codedImg, vector<string> &label);
};

class MultiSourceTsvRawDataFile: public ITsvDataFile{
public:
    void Close();
	void ShuffleData(const string &filename);

	bool IsEOF();
	void MoveToFirst();
	void MoveToLine(int lineNo);
    void MoveToNext();
	int TotalLines();

	// read lines to batch for parallel base64 decoding and image resizing.
	int ReadNextLine(vector<string> &base64codedImg, vector<string> &label);

public: 
    void Open(const vector<string> &fileNames, 
            const vector<bool> &cache_all,
            const vector<int> &colData, const vector<int> &colLabel);
    ~MultiSourceTsvRawDataFile();

private:
    void EnsureShuffleDataInitialized();
private:
    vector<string> _fileNames;
    vector<int> _colData;
    vector<int> _colLabel;

    vector<shared_ptr<ITsvDataFile>> _dataFiles;
    // the first one is the index of the tsv file and the second is the index
    // of the within that file
    std::vector<std::pair<int64_t, int64_t>> _shuffleLines;
	int _currentLine;
};

}
