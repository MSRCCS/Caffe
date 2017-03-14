#include "gtest/gtest.h"

#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/tsv_data_io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class TsvDataIoTest : public ::testing::Test 
{
protected:
  virtual void SetUp()
  {
    MakeTempDir(&source_);
    source_ += "/test.tsv";
    string lineidx_file = ChangeFileExtension(source_, "lineidx");
    string shuffle_file = ChangeFileExtension(source_, "shuffle");
    LOG(INFO) << "Using temporary tsv file " << source_;
    
    FILE *fp_tsv = fopen(source_.c_str(), "w");
    CHECK(fp_tsv) << "Cannot create temporary tsv file " << source_;
    FILE *fp_lineidx = fopen(lineidx_file.c_str(), "w");
    CHECK(fp_lineidx) << "Cannot create lineidx file " << lineidx_file;
    for (int i = 0; i < 10; i++)
    {
      fprintf(fp_lineidx, "%ld\n", ftell(fp_tsv));
      fprintf(fp_tsv, "%d\t%c\t%c\n", i, 'a' + i, 'A' + i);
    }
    fclose(fp_lineidx);
    fclose(fp_tsv);

    FILE *fp_shuffle = fopen(shuffle_file.c_str(), "w");
    CHECK(fp_shuffle) << "Cannot create shuffle file " << shuffle_file;
    for (int i = 0; i < 10; i++)
      fprintf(fp_shuffle, "%d\n", 9 - i);
    fclose(fp_shuffle);
  }

  virtual ~TsvDataIoTest() { }

  void SequentialRead(bool cache_all)
  {
    TsvRawDataFile tsv;
    tsv.Open(this->source_.c_str(), cache_all, 2, 0);
    for (int n = 0; n < 3; n++)
    {
      vector<std::string> base64coded_data;
      vector<std::string> label;
      for (int i = 0; i < 100; i++)
      {
        if (tsv.ReadNextLine(base64coded_data, label) < 0)
          break;
        string data = base64coded_data[i];
        string lbl = label[i];
        string _lbl = "0";
        _lbl[0] += i;
        string _data = "A";
        _data[0] += i;
        EXPECT_EQ(data, _data);
        EXPECT_EQ(lbl, _lbl);
      }
      EXPECT_TRUE(tsv.IsEOF());
      tsv.MoveToFirst();
    }
  }

  void RandomRead(bool cache_all)
  {
    TsvRawDataFile tsv;
    tsv.Open(this->source_.c_str(), cache_all, 2, 0);
    tsv.ShuffleData(ChangeFileExtension(this->source_, "shuffle"));
    EXPECT_EQ(tsv.TotalLines(), 10);
    for (int n = 0; n < 3; n++)
    {
      vector<std::string> base64coded_data;
      vector<std::string> label;
      for (int i = 0; i < 100; i++)
      {
        if (tsv.ReadNextLine(base64coded_data, label) < 0)
          break;
        string data = base64coded_data[i];
        string lbl = label[i];
        string _lbl = "0";
        _lbl[0] += 9 - i;
        string _data = "A";
        _data[0] += 9 - i;
        EXPECT_EQ(data, _data);
        EXPECT_EQ(lbl, _lbl);
      }
      EXPECT_TRUE(tsv.IsEOF());
      tsv.MoveToFirst();
    }
  }

  string source_;
};


TEST_F(TsvDataIoTest, TestSequentialRead)
{
  SequentialRead(false);
}

TEST_F(TsvDataIoTest, TestRandomRead)
{
  RandomRead(false);
}

TEST_F(TsvDataIoTest, TestCachedSequentialRead)
{
  SequentialRead(true);
}

TEST_F(TsvDataIoTest, TestCachedRandomRead)
{
  RandomRead(true);
}

}  // namespace caffe
