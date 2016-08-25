#include "caffe/util/tsv_data_io.hpp"

#ifndef __APPLE__

namespace caffe {

void TextFile::load_buffer()
{
	_bytesInBuffer = fread(&_buffer[0], 1, _buffer.size(), _fp);
	_bufPos = 0;
}

TextFile::TextFile()
{
	_bufPos = 0;
	_bytesInBuffer = 0;
	_fp = NULL;
}

TextFile::~TextFile()
{
	Close();
}

bool TextFile::IsEOF()
{
	return feof(_fp) != 0;
}

int TextFile::Open(const char *fname, int buffer_size)
{
	_fp = fopen(fname, "r");
	if (_fp == NULL)
		return -1;
	_buffer.resize(buffer_size);
	return 0;
}

void TextFile::Close()
{
	if (_fp)
		fclose(_fp);
}

void TextFile::Seek(int64_t pos)
{
#ifdef _MSC_VER
	_fseeki64(_fp, pos, SEEK_SET);
#else
	fseeko64(_fp, pos, SEEK_SET);
#endif
	_bufPos = 0;
	_bytesInBuffer = 0;
}

bool TextFile::ReadLine(string &line)
{
	if (feof(_fp))
		return false;

	line.clear();
	while (1)
	{
		if (_bufPos >= _bytesInBuffer)
		{
			load_buffer();
			if (_bytesInBuffer == 0)
				break;
		}
		char *buf_start = &_buffer[0] + _bufPos;
		char *p = (char *)memchr(buf_start, '\n', _bytesInBuffer - _bufPos);
		if (p == NULL)
		{
			line.append(buf_start, _bytesInBuffer - _bufPos);
			_bufPos = _bytesInBuffer;
		}
		else
		{
			line.append(buf_start, p - buf_start);
			_bufPos = p - &_buffer[0] + 1;	// skip 1 for '\n'
			break;
		}
	}
	return true;
}

TsvRawDataFile::~TsvRawDataFile()
{
	Close();
}

// load line index file to vector lineIndex 
int TsvRawDataFile::LoadLineIndex(const char *fileName, vector<int64_t> &lineIndex)
{
	std::ifstream index_file;
	index_file.open(fileName);
	if (index_file.fail())
	{
		char *err_msg = strerror(errno);
		LOG(FATAL) << "File open failed: " << fileName << ", error: " << err_msg;
		return -1;
	}

	lineIndex.clear();
	while (!index_file.eof())
	{
		std::string line;
		std::getline(index_file, line);
		if (line.length() == 0)
			break;

		lineIndex.push_back(atoll(line.c_str()));
	}

	index_file.close();

	return 0;
}

string TsvRawDataFile::ChangeFileExtension(string filename, string ext)
{
	std::string new_filename = filename;
	size_t result = new_filename.find_last_of('.');

	// Does new_filename.erase(std::string::npos) working here in place of this following test?
	if (std::string::npos != result)
		new_filename.erase(result + 1);

	// append extension:
	new_filename.append(ext);

	return new_filename;
}

int TsvRawDataFile::Open(const char *fileName, int colData, int colLabel)
{
	_tsvFileName = fileName;
	_colData = colData;
	_colLabel = colLabel;

	int err = _dataFile.Open(fileName, 128 * 1024);
	if (err)
	{
		LOG(FATAL) << "TSV file open failed: " << fileName;
		return -1;
	}

	_currentLine = 0;

	return 0;
}

void TsvRawDataFile::Close()
{ 
	_dataFile.Close();
}

void TsvRawDataFile::ShuffleData(string filename)
{
	LOG(INFO) << "Loading idx file...";
	LoadLineIndex(ChangeFileExtension(_tsvFileName, "lineidx").c_str(), _lineIndex);

    string shuffleFile = filename;
    if (shuffleFile.size() == 0)
        shuffleFile = ChangeFileExtension(_tsvFileName, "shuffle");

    LOG(INFO) << "Loading shuffle file...";
	// shuffle file consists of random line numbers (not line index which corresponds to file position of each line)
	// this kind of shuffle is useful for constructing pairwise or triplet data
	// note that the shuffle file may contains more lines than index file, which means that data could be repeatedly used
	// for triplet training.
	vector<int64_t> shuffle;
	LoadLineIndex(shuffleFile.c_str(), shuffle);
	vector<int64_t> newLineIndex(shuffle.size());
	for (int i = 0; i < shuffle.size(); i++)
		newLineIndex[i] = _lineIndex[shuffle[i]];
	_lineIndex = newLineIndex;
}

bool TsvRawDataFile::IsEOF()
{
	return ((_lineIndex.size() == 0 && _dataFile.IsEOF()) || (_lineIndex.size() > 0 && _currentLine >= _lineIndex.size()));
}

int TsvRawDataFile::ReadNextLine(vector<string> &base64codedImg, vector<string> &label)
{
	if (IsEOF())
		return -1;

	if (_lineIndex.size() > 0)
		_dataFile.Seek(_lineIndex[_currentLine]);

	std::string line;
	_dataFile.ReadLine(line);
	if (line.length() == 0)
	{
		_currentLine++;
		return -1;
	}

	// the following block is to directly use data in the string buffer for fast speed.
	vector<int64_t> cell_pos_list;
	size_t line_size = line.size();
	char *ptr = &line[0];
	char *ptr_end = ptr + line_size;
	cell_pos_list.push_back(0);
	for (char *ptr = &line[0]; ptr < ptr_end; ptr++)
	{
		if (*ptr == '\t')
		{
			cell_pos_list.push_back(ptr - &line[0]);   
			cell_pos_list.push_back(ptr - &line[0] + 1);
		}
	}
	cell_pos_list.push_back(line_size); //cell position, 0th: position the image string starts; 1st: position the image string ends; 2nd: position the label string starts; 3rd: position the label string ends; etc...
	int result_num = (int)cell_pos_list.size() / 2;

	if (_colData >= 0)
	{
		CHECK_GT(result_num, _colData) << "colData is out of range of TSV columns.";
		base64codedImg.push_back(line.substr(cell_pos_list[_colData * 2], cell_pos_list[_colData * 2 + 1] - cell_pos_list[_colData * 2]));
	}

	if (_colLabel >= 0)
	{
		CHECK_GT(result_num, _colLabel) << "colLabel is out of range of TSV columns.";
		label.push_back(line.substr(cell_pos_list[_colLabel * 2], cell_pos_list[_colLabel * 2 + 1] - cell_pos_list[_colLabel * 2]));
	}

	_currentLine++;

	return 0;
}

void TsvRawDataFile::MoveToFirst()
{
	_currentLine = 0;
	if (_lineIndex.size() > 0)
		_dataFile.Seek(_lineIndex[_currentLine]);
	else
		_dataFile.Seek(0);
}

void TsvRawDataFile::MoveToLine(int lineNo)
{
	CHECK_GT(_lineIndex.size(), 0) << "Tsv file without .lineidx cannot be randomly accessed.";
	CHECK_GT(_lineIndex.size(), lineNo) << "LineNo (" << lineNo << ") cannot exceed the total line size (" << _lineIndex.size() << ").";
	_currentLine = lineNo;
	_dataFile.Seek(_lineIndex[_currentLine]);
}

int TsvRawDataFile::TotalLines()
{
	CHECK_GT(_lineIndex.size(), 0) << "Tsv file without .lineidx cannot use TotalLines.";
	return _lineIndex.size();
}

}

#endif
