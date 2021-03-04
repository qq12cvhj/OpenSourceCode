

public class DspUtils {
    private static final long MAX_BLOCKS = (long) Math.pow(2.0, 18.0);
    public static INDArray[] stft(INDArray input, int n_fft, int hop_length, int win_length, String windowName) {
        if (n_fft <= 0) throw new UnsupportedOperationException("The n_fft must be > 0");
        if (win_length <= 0) win_length = n_fft;
        if (hop_length <= 0) hop_length = win_length / 4;
        INDArray window;
        WindowFunction windowFunction = new WindowFunction();
        INDArray y_frames;
        float[] returnedArray = null;
        INDArray[] complexDatas = new INDArray[2];
        switch (windowName) {
            case "hann":
                windowFunction.setWindowType(2); //汉宁窗
                double[] windowArray = windowFunction.generate(win_length);
                double[] librosaWindowArray = new double[win_length];
                librosaWindowArray[0] = 0.0;
                for (int i = 0; i < win_length - 1; i++) {
                    librosaWindowArray[i + 1] = windowArray[i];
                }
                window = Nd4j.createFromArray(dArray2fArray(librosaWindowArray));
                window = padCenter(window, n_fft, 0, "CONSTANT", 0);
                window = window.reshape(n_fft, 1);
                input = Nd4j.pad(input, new int[]{n_fft / 2, n_fft / 2}, Pad.Mode.REFLECT, 0);
                input.setOrder('c');
                y_frames = frame(input, n_fft, hop_length, -1);
                int outShape0 = 1 + n_fft / 2;
                int outShape1 = (int) y_frames.shape()[1];
                INDArray stft_matrix_i = Nd4j.rand(DataType.FLOAT, outShape0, outShape1);
                INDArray stft_matrix_r = Nd4j.rand(DataType.FLOAT, outShape0, outShape1);
                /*stft_matrix_r.setOrder('f');
                stft_matrix_i.setOrder('f');*/
                int n_columns = (int) (MAX_BLOCKS / (outShape0 * stft_matrix_r.dataType().width())) / 2; 
                for (int bl_s = 0; bl_s < outShape1; bl_s += n_columns) {
                    int bl_t = Math.min(bl_s + n_columns, outShape1);
                    for (int i = bl_s; i < bl_t; i++) {
                        INDArray column4rfft = y_frames.getColumn(i).mulColumnVector(window);
                        float[] tmp = new float[(int) (column4rfft.length() * 2)];
                        float[] columnArray = column4rfft.toFloatVector();
                        for (int j = 0; j < column4rfft.length(); j++) {
                            tmp[2 * j] = columnArray[j];
                            tmp[2 * j + 1] = 0;
                        }
                        FloatFFT_1D floatFFT_1D = new FloatFFT_1D(column4rfft.length());
                        floatFFT_1D.complexForward(tmp);
                        float[] tmp2 = new float[columnArray.length / 2 + 1];
                        float[] tmp3 = new float[columnArray.length / 2 + 1];
                        for (int j = 0, k = 0; j < tmp.length / 2 + 1; j += 2, k++) {
                            //数组拷贝，每次在tmp中每两位存储一个实数部分一个虚数部分
                            tmp2[k] = tmp[j];
                            tmp3[k] = tmp[j + 1];
                        }
                        stft_matrix_r.putColumn(i, Nd4j.createFromArray(tmp2));
                        stft_matrix_i.putColumn(i, Nd4j.createFromArray(tmp3));
                    }
                }
                complexDatas[0] = stft_matrix_r;
                complexDatas[1] = stft_matrix_i;
                break;
        }
        return complexDatas;
    }

    private static INDArray padCenter(INDArray data, int size, int axis, String padMode, double padValue) {
        int n = (int) data.shape()[axis];
        int lpad = (size - n) / 2;
        int[][] lengths = new int[data.rank()][2]; //初始化后就是0，所有元素都是0
        lengths[axis][0] = lpad;
        lengths[axis][1] = size - n - lpad;
        if (padMode.equals("CONSTANT")) {
            return Nd4j.pad(data, lengths, Pad.Mode.CONSTANT, padValue);
        } else {
            throw new UnsupportedOperationException("Invalid");
        }
    }

    private static INDArray frame(INDArray input, int frameLength, int hopLength, int axis) {
        if (axis == -1) {
            axis = input.rank() - 1;
        }
        int n_frames = (int) (1 + (input.shape()[axis] - frameLength) / hopLength);
        int itemSize = input.dataType().width();
        List<Long> newStridesList = new ArrayList<>();

        for (long stride : input.stride()) {
            if (stride > 0) newStridesList.add(stride * 8 / 8); 
        }
        long[][] newStrideArray = new long[1][newStridesList.size()];
        for (int i = 0; i < newStridesList.size(); i++) {
            newStrideArray[0][i] = newStridesList.get(i);
        }
        INDArray newStride = Nd4j.prod(Nd4j.createFromArray(newStrideArray)).muli(8);
        List<Integer> shapeList = new LinkedList<>();
        List<Integer> strideList = new LinkedList<>();
        if ((input.ordering() == 'c')) {
            if (axis != input.rank() - 1) {
                throw new UnsupportedOperationException("the input order is c but the axis should be -1(the last dim)");
            }
            for (int i = 0; i < input.rank() - 1; i++) {
                shapeList.add((int) input.shape()[i]);
            }
            shapeList.add(frameLength);
            shapeList.add(n_frames);
            for (long stride : input.stride()) {
                strideList.add((int) stride * 8);
            }
            for (int stride : newStride.toIntVector()) {
                strideList.add(stride * hopLength);
            }
        } else if ((input.ordering() == 'f')) {
            if (axis != 0) {
                throw new UnsupportedOperationException("the input order is f but the axis should be 0(the first dim)");
            }
            shapeList.add(n_frames);
            shapeList.add(frameLength);
            for (int i = 1; i < input.rank(); i++) {
                shapeList.add((int) input.shape()[i]);
            }
            for (int stride : newStride.toIntVector()) {
                strideList.add(stride * hopLength);
            }
            for (long stride : input.stride()) {
                strideList.add((int) stride * 8);
            }
        }
        int[] shapeArray = new int[shapeList.size()];
        int[] strideArray = new int[strideList.size()];
        for (int i = 0; i < shapeList.size(); i++) {
            shapeArray[i] = shapeList.get(i);
        }
        for (int i = 0; i < strideList.size(); i++) {
            strideArray[i] = strideList.get(i) / 8;
        }
        input.setShapeAndStride(shapeArray, strideArray);

        return input;
    }

    public static float[] dArray2fArray(double[] doubles) {
        int len = doubles.length;
        float[] floats = new float[len];
        for (int i = 0; i < len; i++) {
            floats[i] = (float) doubles[i];
        }
        return floats;
    }

    private static INDArray pad_and_partition(INDArray tensor, int length) {
        int oldsize = (int) (tensor.shape()[3]);
        int newSize = (int) Math.ceil((double) oldsize / length) * length;
        int[][] padWidthArray = new int[tensor.rank()][2];
        padWidthArray[tensor.rank() - 1][0] = 0;
        padWidthArray[tensor.rank() - 1][1] = newSize - oldsize;
        INDArray padTensor = Nd4j.pad(tensor, padWidthArray);
        int arrayLen = (int) Math.ceil((float) padTensor.shape()[3] / length);
        INDArray[] indArrays = new INDArray[arrayLen];
        for (int i = 0; i < arrayLen; i++) {
            indArrays[i] = padTensor.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(i * length, (i + 1) * length));
        }
        return Nd4j.concat(0, indArrays);
    }

    /**
     * 将大于一定频率的数据砍掉，只保留不超过maxFreq的值
     *
     * @param tensor  要过滤的数据
     * @param maxFreq 最大频率
     * @return 一个新的tensor, 转换过的
     */
    private static INDArray filterStftResults(INDArray tensor, int maxFreq) {
        //输入的tensor,根据实际需求，tensor为2×F×L×1或者2×F×L×2，
        // F为频率，L为长度，前面的2为声道，后面的1或2为实数部分或者虚数部分
        return tensor.get(NDArrayIndex.interval(0, maxFreq), NDArrayIndex.all());
    }

    public static INDArray[] computeStftAndMagnitude(INDArray wavTensor) {
        INDArray first_channel = wavTensor.getRow(0);
        INDArray second_channel = wavTensor.getRow(1);
        INDArray[] indArrays_first = stft(first_channel, 4096, 1024, 4096, "hann");
        Log.d(TAG, "indArrays_first stft结束");
        INDArray[] indArrays_second = stft(second_channel, 4096, 1024, 4096, "hann");
        Log.d(TAG, "indArrays_second stft结束");
        INDArray realPart_first = filterStftResults(indArrays_first[0], 1024);
        INDArray realPart_second = filterStftResults(indArrays_second[0], 1024);
        INDArray imgPart_first = filterStftResults(indArrays_first[1], 1024);
        INDArray imgPart_second = filterStftResults(indArrays_second[1], 1024);
        INDArray magnitude_first = sqrt(realPart_first).add(sqrt(imgPart_first));
        INDArray magnitude_second = sqrt(realPart_second).add(sqrt(imgPart_second));
        INDArray realPart = Nd4j.stack(0, realPart_first, realPart_second);
        INDArray imgPart = Nd4j.stack(0, imgPart_first, imgPart_second);
        INDArray magnitude = sqrt(Nd4j.stack(0, magnitude_first, magnitude_second));
        return new INDArray[]{Nd4j.stack(3, realPart, imgPart), magnitude};

    }

    /**
     * 获得神经网络能用的输入tensor
     *
     * @return 一个能给模型使用的tensor
     */
    public static INDArray getInput4Net(INDArray input) {
        input.permutei(3, 0, 1, 2);
        input = pad_and_partition(input, 512);
        input.permutei(0, 1, 3, 2);
        return input;
    }


    private static INDArray inverseStft(int winLength, INDArray stftTensor) {
        int pad = (int) (winLength / 2 + 1 - stftTensor.shape()[1]);
        int[][] padLengths = new int[stftTensor.rank()][2];
        padLengths[1][1] = pad;
        stftTensor = Nd4j.pad(stftTensor, padLengths);
        //执行真正的istft算法
        return istft(stftTensor, 4096, 1024);
    }

    //仿照librosa的istft算法

    /**
     * @param stftTensor 输入矩阵
     * @param winLength  窗口
     * @param hopLength  间距
     * @return 一个x维张量.x为声道数。如，立体声，有左右两个声道，那么返回的就是2维张量。x为stftTensor[0]
     */
    public static INDArray istft(INDArray stftTensor, int winLength, int hopLength) {
        int channel = (int) stftTensor.shape()[0];
        INDArray[] wavTensors = new INDArray[channel];
        for (int c = 0; c < channel; c++) {
            INDArray singleChannelTensor = stftTensor.slice(c, 0); //取第c个channel
            int n_fft = (int) (2 * (singleChannelTensor.shape()[0] - 1));
            INDArray window;
            WindowFunction windowFunction = new WindowFunction();
            windowFunction.setWindowType(2);//2是汉宁窗
            double[] windowArray = windowFunction.generate(winLength);
            double[] librosaWindowArray = new double[winLength];
            //跟librosa得出的window结果有些不一样，偏了一位，这里处理一下.
            librosaWindowArray[0] = 0.0;
            /*for (int i = 0; i < winLength; i++) {
                librosaWindowArray[i] = windowArray[i];
            }*/
            for (int i = 0; i < winLength - 1; i++) {
                librosaWindowArray[i + 1] = windowArray[i];
            }
            window = Nd4j.createFromArray(dArray2fArray(librosaWindowArray));
            window = padCenter(window, n_fft, 0, "CONSTANT", 0); //可能有问题
            window = window.reshape(window.length(), 1);
            int n_frames = (int) singleChannelTensor.shape()[1];
            int expected_signal_len = n_fft + hopLength * (n_frames - 1); //输出的数组长度
            INDArray y = Nd4j.zeros(DataType.FLOAT, expected_signal_len);
            int n_columns = (int) (MAX_BLOCKS / (singleChannelTensor.shape()[0] * singleChannelTensor.dataType().width())) / 2; //可能有问题
            int frame = 0;
            for (int bls = 0; bls < n_frames; bls += n_columns) {
                int blt = Math.min(bls + n_columns, n_frames);
                INDArray tmpTensor = irfft(singleChannelTensor.get(NDArrayIndex.all(), NDArrayIndex.interval(bls, blt)), winLength);
                INDArray ytmp = window.mul(tmpTensor);
                __overlap_add(y, frame * hopLength, ytmp, hopLength);
                frame += blt - bls;
            }
            INDArray ifft_window_sum = window_sumsquare(window, n_frames, hopLength, winLength, n_fft);
            for (int i = 0; i < ifft_window_sum.length(); i++) {
                if (ifft_window_sum.getFloat(i) > Float.MIN_VALUE) {
                    y.putScalar(i, y.getFloat(i) / ifft_window_sum.getFloat(i));
                }
            }
            y = y.get(NDArrayIndex.interval(n_fft / 2, y.length() - n_fft / 2));
            wavTensors[c] = y;
        }
        return Nd4j.stack(0, wavTensors);
    }

    /**
     * 执行irfft变换
     *
     * @param input 输入张量，结构为m*n*2;通过第2维切片，获得2个m*n的矩阵，第一个为实数阵，第二个为虚数阵。
     * @return 做完变换后的张量
     */
    private static INDArray irfft(INDArray input, int winLen) {
        long columnCount = input.shape()[1];
        INDArray realPart = input.slice(0, 2);
        INDArray imgPart = input.slice(1, 2);
        INDArray indArray = Nd4j.create(DataType.FLOAT, winLen, columnCount);
        for (int i = 0; i < columnCount; i++) {
            float[] realVec = realPart.getColumn(i).toFloatVector();
            float[] imgVec = imgPart.getColumn(i).toFloatVector();
            float[] array4irfft = new float[winLen * 2];
            for (int k = 0; k < realVec.length; k++) {
                array4irfft[2 * k] = realVec[k];
                array4irfft[2 * k + 1] = imgVec[k];
            }
            //要满足对称性
            for (int k = 0; k < winLen / 2; k++) {
                array4irfft[2 * (winLen / 2 + k)] = array4irfft[2 * (winLen / 2 - k)];
                array4irfft[2 * (winLen / 2 + k) + 1] = -array4irfft[2 * (winLen / 2 - k) + 1];
            }
            FloatFFT_1D floatFFT_1D = new FloatFFT_1D(winLen);
            floatFFT_1D.complexInverse(array4irfft, true);
            float[] array = new float[winLen];
            for (int j = 0; j < winLen; j++) {
                array[j] = array4irfft[2 * j];
            }
            indArray.putColumn(i, Nd4j.createFromArray(array));
        }
        return indArray;
    }

    private static INDArray window_sumsquare(INDArray window, int n_frames, int hopLength, int winLength, int n_fft) {
        int n = n_fft + hopLength * (n_frames - 1);
        INDArray x = Nd4j.zeros(DataType.FLOAT, n);
        window.muli(window);
        window = padCenter(window, n_fft, 0, "CONSTANT", 0).reshape(winLength);
        return __window_ss_fill(x, window, n_frames, hopLength);
    }

    private static INDArray __window_ss_fill(INDArray x, INDArray win_sq, int n_frames, int hop_length) {
        int n = (int) x.length();
        int n_fft = (int) win_sq.length();
        for (int i = 0; i < n_frames; i++) {
            int sample = i * hop_length;
            int tmp_left = Math.min(n, sample + n_fft);
            int tmp_right = Math.max(0, Math.min(n_fft, n - sample));
            INDArrayIndex tmpIndex4Add = NDArrayIndex.interval(sample, tmp_left);
            INDArrayIndex tmpIndex4Add2 = NDArrayIndex.interval(0, tmp_right);
            x.get(NDArrayIndex.interval(sample, tmp_left)).addi(win_sq.get(NDArrayIndex.interval(0, tmp_right)));
        }
        return x;
    }

    private static void __overlap_add(INDArray y, int start, INDArray ytmp, int hopLength) {
        int n_fft = (int) ytmp.shape()[0];
        for (int frame = 0; frame < ytmp.shape()[1]; frame++) {
            int sample = frame * hopLength;
            INDArray tmp = ytmp.get(NDArrayIndex.all(), NDArrayIndex.point(frame));
            INDArray beAdded = y.get(NDArrayIndex.interval(sample + start, sample + n_fft + start));
            beAdded.addi(tmp);
        }
    }

    public static INDArray[] getOutputWavTensors(INDArray[] inputTensors, INDArray stftTensor, int stftSize) {
        INDArray mask0 = inputTensors[0];
        INDArray mask1 = inputTensors[1];
        INDArray mask_sum = mask0.mul(mask0).add(mask1.mul(mask1)).sum(0).add(1e-10f);
        INDArray[] outputTensors = new INDArray[2];
        for (int j = 0; j < 2; j++) {
            INDArray inputTensor = inputTensors[j];
            INDArray mask = inputTensor.mul(inputTensor).add(2 * (1e-10f) / 2).div(mask_sum);
            mask.permutei(0, 1, 3, 2);
            int splitSize = (int) mask.shape()[0];
            INDArray[] tmp = new INDArray[splitSize];
            for (int i = 0; i < splitSize; i++) {
                tmp[i] = mask.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            }
            mask = Nd4j.concat(2, tmp);
            mask = Nd4j.stack(3, mask);
            mask = mask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, stftSize), NDArrayIndex.all());
            INDArray stft_masked = stftTensor.mul(mask);
            INDArray outputTensor = inverseStft(4096, stft_masked);
            outputTensors[j] = outputTensor;
        }
        return outputTensors;
    }
  
}
