#include <utility>
#include <boost/algorithm/string.hpp>
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
    fclose(fp);
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

void LoadLineIndex(const string &filename, vector<std::pair<int64_t, int64_t>>&lineIndex) {
	std::ifstream index_file;
	index_file.open(filename.c_str());
    CHECK(!index_file.fail()) << "File open failed: " << filename 
        << ", error: " << strerror(errno);

	lineIndex.clear();
	while (!index_file.eof()) {
		std::string line;
		std::getline(index_file, line);
		if (line.length() == 0) {
			break;
        }
        vector<string> parts;
        boost::split(parts, line, boost::is_any_of("\t"));
        if (parts.size() == 1) {
            lineIndex.push_back(std::make_pair<int64_t, int64_t>(0L, atoll(parts[0].c_str())));
        } else if (parts.size() == 2) {
		    lineIndex.push_back(std::make_pair<int64_t, int64_t>(atoll(parts[0].c_str()), 
                        atoll(parts[1].c_str())));
        } else {
            LOG(FATAL) << "illegal line: " << line;
        }
	}

    LOG(INFO) << "# of idx loaded: " << lineIndex.size();

	index_file.close();
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

void TsvRawDataFile::ShuffleData(const string &filename)
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
	return _currentLine >= TotalLines();
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
    if (lineNo < TotalLines())
    {
        if (_shuffleLines.size() > 0)
            _dataFile.Seek(_lineIndex[_shuffleLines[lineNo]]);
        else
            _dataFile.Seek(_lineIndex[lineNo]);
    }
    // if lineNo >= TotalLines(), we just set _currentLine. IsEOF() can detect this status without problem.
    _currentLine = lineNo;
}

void TsvRawDataFile::MoveToNext()
{
    if (!IsEOF())
        MoveToLine(_currentLine + 1);
}

int TsvRawDataFile::TotalLines()
{
    return _shuffleLines.size() > 0 ? _shuffleLines.size() : _lineIndex.size();
}

ITsvDataFile* ITsvDataFile::make_tsv(const char* fileName, bool cache_all, int colData, int colLabel) {
    TsvRawDataFile* result = new TsvRawDataFile();
    auto code = result->Open(fileName, cache_all, colData, colLabel);
    CHECK(!code);
    return result;
}

ITsvDataFile* ITsvDataFile::make_tsv(const vector<string> &fileNames, 
            const vector<bool> &cache_all,
            const vector<int> &colData,
            const vector<int> &colLabel) {
    CHECK_EQ(fileNames.size(), cache_all.size());
    CHECK_EQ(fileNames.size(), colData.size());
    CHECK_EQ(fileNames.size(), colLabel.size());
    if (fileNames.size() == 1) {
        return make_tsv(fileNames[0].c_str(), 
                cache_all[0], colData[0], colLabel[0]);
    } else {
        auto result = new MultiSourceTsvRawDataFile();
        result->Open(fileNames, cache_all, colData, colLabel);
        return result;
    }
}

ITsvDataFile* ITsvDataFile::make_tsv(const vector<string> &fileNames, 
            bool cache_all,
            int colData,
            int colLabel) {
    vector<int> vec_colData(fileNames.size(), colData);
    vector<int> vec_colLabel(fileNames.size(), colLabel);
    vector<bool> vec_cacheAll(fileNames.size(), cache_all);
    return ITsvDataFile::make_tsv(fileNames, vec_cacheAll, vec_colData, vec_colLabel);
}

void MultiSourceTsvRawDataFile::Open(const vector<string> &fileNames, 
        const vector<bool> &cache_all,
        const vector<int> &colData, const vector<int> &colLabel) {
    CHECK_EQ(fileNames.size(), colData.size());
    CHECK_EQ(fileNames.size(), colLabel.size());

    int num = fileNames.size();
    _dataFiles.clear();
    for (int i = 0; i < num; i++) {
        shared_ptr<ITsvDataFile> tsvFile(ITsvDataFile::make_tsv(fileNames[i].c_str(), cache_all[i], colData[i], colLabel[i]));
        _dataFiles.push_back(tsvFile);
    }

	_currentLine = 0;
}

void MultiSourceTsvRawDataFile::Close() {
    for (auto f: _dataFiles) {
        f->Close();
    }
}

void MultiSourceTsvRawDataFile::ShuffleData(const string &filename) {
    LOG(INFO) << "Loading shuffle file...";
    LoadLineIndex(filename, _shuffleLines); 
}

bool MultiSourceTsvRawDataFile::IsEOF() {
	return _currentLine >= TotalLines();
}

void MultiSourceTsvRawDataFile::EnsureShuffleDataInitialized() {
    if (_shuffleLines.size() == 0) {
        for (int64_t i = 0; i < _dataFiles.size(); i++) {
            for (int64_t j = 0; j < _dataFiles[i]->TotalLines(); j++) {
                _shuffleLines.emplace_back(i, j);
            }
        }
    }
}

void MultiSourceTsvRawDataFile::MoveToFirst() {
    this->MoveToLine(0);
}

void MultiSourceTsvRawDataFile::MoveToLine(int lineNo) {
    if (lineNo < TotalLines()) {
        CHECK_GE(lineNo, 0);
        CHECK_LT(lineNo, _shuffleLines.size());
        auto &p = _shuffleLines[lineNo];
        CHECK_GE(p.first, 0);
        CHECK_LT(p.first, _dataFiles.size());
        _dataFiles[p.first]->MoveToLine(p.second);
    }
    _currentLine = lineNo;
}

void MultiSourceTsvRawDataFile::MoveToNext() {
    if (!IsEOF()) {
        MoveToLine(_currentLine + 1);
    }
}

int MultiSourceTsvRawDataFile::TotalLines() {
    EnsureShuffleDataInitialized();
    return _shuffleLines.size();
}

int MultiSourceTsvRawDataFile::ReadNextLine(vector<string> &base64codedImg, vector<string> &label) {
	if (IsEOF()) {
		return -1;
    }
    CHECK_GE(_currentLine, 0);
    CHECK_LT(_currentLine, _shuffleLines.size());
    MoveToLine(_currentLine);
    auto &p = _shuffleLines[_currentLine];
    CHECK_GE(p.first, 0);
    CHECK_LT(p.first, _dataFiles.size());
    _dataFiles[p.first]->ReadNextLine(base64codedImg, label);
	_currentLine++;
    return 0;
}

}
