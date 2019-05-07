#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"
#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include "make_input.h"

#define PCLOUD_SIZE 1000000 //the size of buffer for the point cloud

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, std::string calibList)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mCalibList(calibList)
    {
        //get calib file names
        std::fstream fin(mCalibList);
        std::string ReadLine;
        while(std::getline (fin, ReadLine))
        {
            mFileList.push_back(ReadLine);
        }
        //set network input dims
        mDims = nvinfer1::DimsNCHW{1, 3, 200, 400};
        mImageSize = mDims.c() * mDims.h() * mDims.w();
        mTruthSize = mDims.h() * mDims.w() ;
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize * mTruthSize, 0);
        mFileBatch.resize(mDims.n() * mImageSize, 0); // one input image buffer
        mFileLabels.resize(mDims.n() * mTruthSize, 0);
        mPointsBuf.resize(PCLOUD_SIZE, 0);
        reset(0);
    }

    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.n();
        skip(firstBatch);
    }

    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize)
        {
            if (!update())
                return false;
            std::copy_n(getFileBatch(), csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels(), csize * mTruthSize, getLabels() + batchPos * mTruthSize);
        }
        mBatchCount++;
        return true;
    }

    void skip(int skipCount)
    {
        mFileCount = skipCount;
    }

    float* getBatch() { return &mBatch[0]; }
    float* getLabels() { return &mLabels[0]; }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    nvinfer1::DimsNCHW getDims() const { return mDims; }

private:
    float* getFileBatch() { return &mFileBatch[0]; }
    float* getFileLabels() { return &mFileLabels[0]; }
    float* getPointsBuf() { return &mPointsBuf[0]; }

    bool update()
    {
        memset(getPointsBuf(), 0, PCLOUD_SIZE*sizeof(float));
        memset(getFileBatch(), 0, mImageSize*sizeof(float));
        memset(getFileLabels(), 0, mTruthSize*sizeof(float));
        char buf_s[64];
        int size = 0;
        mFileCount++;
        std::string inputFileName = mFileList[mFileCount];
        //std::cout<<inputFileName<<std::endl;
        FILE* file = fopen(inputFileName.c_str(), "rb");
        if (!file)
            return false;
        else{
            fseek(file,0L,SEEK_END); 
            size = ftell(file);
            fseek(file,0L,SEEK_SET);
            fread(getPointsBuf(), 1, size, file);
        }
        int points_num = size / sizeof(float) / 5;
        xyzic_to_image(getPointsBuf(), points_num, -45, 45, -30, 10, 0.225, 0.2, getFileBatch(), getFileLabels());
        fclose(file);
        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};

    int mFileCount{0}, mFileBatchPos{0};
    int mImageSize{0};
    int mTruthSize{0};

    nvinfer1::DimsNCHW mDims;
    std::string mCalibList;
    std::vector<std::string> mFileList;
    std::vector<float> mBatch;
    std::vector<float> mLabels;
    std::vector<float> mFileBatch;
    std::vector<float> mFileLabels;
    std::vector<float> mPointsBuf;//the buffer for the point cloud
};

#endif
