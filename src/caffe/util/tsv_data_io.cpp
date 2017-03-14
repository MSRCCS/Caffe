#include "caffe/util/tsv_data_io.hpp"

namespace caffe {

string ChangeFileExtension(string filename, string ext)
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

int64_t GetFileSize(std::string filename)
{
    int64_t pos;
    FILE *fp = fopen(filename.c_str(), "rb");
#ifdef _MSC_VER
    _fseeki64(fp, 0, SEEK_END);
    pos = _ftelli64(fp);
#else
    fseeko64(fp, 0, SEEK_END);
    pos = ftello64(fp);
#endif
    return pos;
}

void TextFile::load_buffer()
{
  // if eof is met, _bytesInBuffer will be 0.
	_bytesInBuffer = fread(&_buffer[0], 1, _buffer.size(), _fp);
	_bufPos = 0;
}

TextFile::TextFile()
{
    _cacheAll = false;
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
  if (_cacheAll)
    return _bufPos == _bytesInBuffer;

	return _bytesInBuffer == 0;
}

int TextFile::Open(const char *fname, bool cache_all)
{
	_fp = fopen(fname, "rb");
	if (_fp == NULL)
		return -1;
    _cacheAll = cache_all;
    int64_t file_size = GetFileSize(fname);
    //if (file_size < 1 * 1024 * 1024 * 1024) // for file size less than 1GB, default to cache all
    //  _cacheAll = true;
    int64_t buffer_size = _cacheAll ? file_size : 10 * 1024;
    _buffer.resize(buffer_size);
    if (_cacheAll)
        LOG(INFO) << "Caching file: " << fname;
    load_buffer();
	return 0;
}

void TextFile::Close()
{
    if (_fp)
    {
        fclose(_fp);
        _fp = NULL;
    }
}

void TextFile::Seek(int64_t pos)
{
    if (_cacheAll)
    {
        _bufPos = pos;
    }
    else
    {
#ifdef _MSC_VER
        _fseeki64(_fp, pos, SEEK_SET);
#else
        fseeko64(_fp, pos, SEEK_SET);
#endif
        _bufPos = 0;
        _bytesInBuffer = -1;
    }
}

bool TextFile::ReadLine(string &line)
{
	if (IsEOF())
		return false;

	line.clear();
	while (1)
	{
		if (_bufPos >= _bytesInBuffer)
		{
			load_buffer();
			if (IsEOF())
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
            int extra_char = *(p - 1) == '\r' ? 1 : 0;
            line.append(buf_start, p - extra_char - buf_start);
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

int TsvRawDataFile::Open(const char *fileName, bool cache_all, int colData, int colLabel)
{
	_tsvFileName = fileName;
	_colData = colData;
	_colLabel = colLabel;

  int err = _dataFile.Open(fileName, cache_all);
	if (err)
	{
		LOG(FATAL) << "TSV file open failed: " << fileName;
		return -1;
	}

	_currentLine = 0;

    LOG(INFO) << "Loading idx file...";
    LoadLineIndex(ChangeFileExtension(_tsvFileName, "lineidx").c_str(), _lineIndex);

    return 0;
}

void TsvRawDataFile::Close()
{ 
	_dataFile.Close();
}

void TsvRawDataFile::ShuffleData(string filename)
{
    // shuffle file consists of random line numbers (not line index which corresponds to file position of each line)
    // this kind of shuffle is useful for constructing pairwise or triplet data
    // note that the shuffle file may contains more lines than index file, which means that data could be repeatedly used
    // for triplet training.
    LOG(INFO) << "Loading shuffle file...";
	LoadLineIndex(filename.c_str(), _shuffleLines);
}

bool TsvRawDataFile::IsEOF()
{
	return ((_shuffleLines.size() == 0 && _dataFile.IsEOF()) || (_shuffleLines.size() > 0 && _currentLine >= _shuffleLines.size()));
}

int TsvRawDataFile::ReadNextLine(vector<string> &base64codedImg, vector<string> &label)
{
	if (IsEOF())
		return -1;

	if (_shuffleLines.size() > 0)
		_dataFile.Seek(_lineIndex[_shuffleLines[_currentLine]]);

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
	if (_shuffleLines.size() > 0)
		_dataFile.Seek(_lineIndex[_shuffleLines[_currentLine]]);
	else
		_dataFile.Seek(0);
}

void TsvRawDataFile::MoveToLine(int lineNo)
{
	CHECK_GT(_lineIndex.size(), 0) << "Tsv file without .lineidx cannot be randomly accessed.";
	CHECK_GT(_shuffleLines.size(), lineNo) << "LineNo (" << lineNo << ") cannot exceed the total line size (" << _shuffleLines.size() << ").";
	_currentLine = lineNo;
	_dataFile.Seek(_lineIndex[_shuffleLines[_currentLine]]);
}

int TsvRawDataFile::TotalLines()
{
	CHECK(_shuffleLines.size() > 0 || _lineIndex.size() > 0) << "Tsv file without .shuffle or .lineidx cannot use TotalLines.";
    if (_shuffleLines.size() > 0)   // use .shuffle to get total lines
	    return _shuffleLines.size();
    return _lineIndex.size();   // use .lineidx to get total lines
}

}
