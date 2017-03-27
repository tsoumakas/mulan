package mulan.evaluation.measure.regression.macro;

import java.util.Arrays;

import mulan.data.MultiLabelInstances;

public abstract class MacroRelativeRegressionMeasureBase extends MacroRegressionMeasureBase {
    /**
     * holds the mean (calculated from train set) prediction's error (total or
     * max) absolute or squared or logarithmic per target
     */
    protected double[] trainMeanPredError;
    /**
     * holds the mean (calculated from full set) prediction's error (total or
     * max) absolute or squared or logarithmic per target
     */
    protected double[] fullMeanPredError;
    /** holds the mean per target in train set */
    protected double[] targetMeansTrain;
    /** holds the mean per target in full dataset */
    protected double[] targetMeansFull;

    public MacroRelativeRegressionMeasureBase(MultiLabelInstances train, MultiLabelInstances test) {
        super(train.getNumLabels());
        int numOfLabels = train.getNumLabels();
        trainMeanPredError = new double[numOfLabels];
        fullMeanPredError = new double[numOfLabels];
        targetMeansTrain = new double[numOfLabels];
        targetMeansFull = new double[numOfLabels];
        int[] labelIndices = train.getLabelIndices();
        for (int i = 0; i < numOfLabels; i++) {
            targetMeansTrain[i] = train.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            double testAverage = test.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            int trainInstances = train.getDataSet().numInstances();
            int testInstances = test.getDataSet().numInstances();
            int allInstances = trainInstances + testInstances;
            targetMeansFull[i] = (targetMeansTrain[i] * trainInstances + testAverage
                    * testInstances)
                    / allInstances;
        }
    }

    public double getTargetAverageFull(int labelIndex) {
        return targetMeansFull[labelIndex];
    }

    public double getTargetAverageTrain(int labelIndex) {
        return targetMeansTrain[labelIndex];
    }

    public void reset() {
        super.reset();
        Arrays.fill(trainMeanPredError, 0.0);
        Arrays.fill(fullMeanPredError, 0.0);
        // the following should NOT be reset!!!
        // Arrays.fill(targetMeansTrain, 0.0);
        // Arrays.fill(targetMeansFull, 0.0);

    }

}
