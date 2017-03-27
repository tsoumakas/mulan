package mulan.regressor.malsar;

import java.util.Arrays;

import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Common root class for all MALSAR methods
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public abstract class MalsarWrapper extends MultiLabelLearnerBase{

    /**
     * The type of normalization to apply.
     */
    public enum normalizationType {
        /** no normalization */
        none,
        /** z-score normalization */
        zScore,
        /** maximum feature value normalization */
        maxFeatureValue
    }

    /**
     * From MALSAR: "All optimization algorithms in our package are implemented using iterative
     * methods. Users can use the optional opts input to specify starting points, termination
     * conditions, tolerance, and maximum iteration number. The input opts is a structure variable.
     * To specify an option, user can add corresponding fields. If one or more required fields are
     * not specified, or the opts variable is not given, then default values will be used. The
     * default values can be changed in init opts.m in /MALSAR/utils."
     */
    public enum optsInit {
        /**
         * If 0 is specified then the starting points will be initialized to a guess value computed
         * from data. For example, in the least squares loss, the model W(:, i) for i-th task is
         * initialized by X{i} * Y{i}.
         */
        zero(0),
        /**
         * If 1 is specified then opts.W0 is used. Note that if value 1 is specified in .init but
         * the field .W0 is not specified, then .init will be forced to the default value.
         */
        one(1),
        /** (default). If 2 is specified, then the starting point will be a zero matrix. */
        two(2);

        private final int value;

        private optsInit(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    /**
     * From MALSAR: "In this package, there are 4 types of termination conditions supported for all
     * optimization algorithms."
     */
    public enum optsTFlag {
        zero(0), /** (default). */
        one(1), two(2), three(3);

        private final int value;

        private optsTFlag(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    /**
     * The names that should be used for the X/Y cell arrays and the parameters array in Matlab.
     */
    protected final String xCellArrayMatName = "X";

    protected final String yCellArrayMatName = "Y";;

    protected final String paramArrayMatName = "W";;

    protected final String optsMatName = "opts";;

    protected normalizationType normalization = normalizationType.none;

    /* defining optimization options and setting their (MALSAR) default values */
    protected optsInit optInit = optsInit.zero;

    protected optsTFlag optTFlag = optsTFlag.one;

    /** tolerance */
    protected double optTol = 0.0001;
    /**
     * From MALSAR: "When the tolerance and/or termination condition is not properly set, the
     * algorithms may take an unacceptable long time to stop. In order to prevent this situation,
     * users can provide the maximum number of iterations allowed for the solver, and the algorithm
     * stops when the maximum number of iterations is achieved even if the termination condition is
     * not satisfied."
     */
    protected int optMaxIter = 1000;
    /** whether a bias column should be added to the data */
    protected boolean addBias = false;
    /**
     * The full path to the root folder of Malsar's Matlab implementation. This is required for
     * adding Malsar to Matlab's path after clearing any existing Matlab variables.
     */
    protected String malsarMatlabPath;

    /**
     * A MatlabProxy used to control Matlab.
     */
    protected MatlabProxy proxy;

    /** the learned parameter matrix (model) */
    protected double[][] W;
    /** feature column means used when z-score normalization is applied */
    protected double[] means;

    /** feature column standard deviations used when z-score normalization is applied */
    protected double[] stds;
    /** stores the maximum value of each feature in the training set */
    protected double[] maxFeatureValues;

    protected int maxNumCompThreads;
    private NominalToBinary nomToBinFilter;
    private ReplaceMissingValues replaceMissingFilter;

    public MalsarWrapper(String malsarPath, MatlabProxy proxy, int maxNumCompThreads) {
        this.malsarMatlabPath = malsarPath;
        this.proxy = proxy;
        this.maxNumCompThreads = maxNumCompThreads;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        Instances dataSet = trainingSet.getDataSet();
        // work on a copy of the dataset
        Instances data = new Instances(dataSet);
        // first handle multi-valued nominal features and features with missing values
        // all targets attributes are numeric, thus this filter should leave them unaffected
        // nomToBinFilter = new NominalToBinary();
        // nomToBinFilter.setInputFormat(data);
        // data = Filter.useFilter(data, nomToBinFilter);
        // after this transformation, the number of features can be taken
        int numFeatures = data.numAttributes() - numLabels;
        // it's ok to replace missing values in target attributes as well
        // or perhaps its better to first delete instances with any of the target values missing
        Instances dataNotMissing = new Instances(data, 0);
        int missingTargetsCounter = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            boolean missingTargets = false;
            for (int j = 0; j < numLabels; j++) {
                if (inst.isMissing(inst.numAttributes() - numLabels + j)) {
                    missingTargets = true;
                    missingTargetsCounter++;
                    break;
                }
            }
            if (!missingTargets) {
                dataNotMissing.add(inst);
            }
        }
        debug("Instances with missing targets " + missingTargetsCounter);
        replaceMissingFilter = new ReplaceMissingValues();
        replaceMissingFilter.setInputFormat(dataNotMissing);
        dataNotMissing = Filter.useFilter(dataNotMissing, replaceMissingFilter);

        // ---------------------------------
        // --- data preparation started ---
        debug("Performing pre-processing");
        double[][] X = new double[dataNotMissing.numInstances()][];
        double[][] Y = new double[dataNotMissing.numInstances()][];
        // the following assumes that all target attributes are in last positions
        for (int i = 0; i < dataNotMissing.numInstances(); i++) {
            double[] vector = dataNotMissing.instance(i).toDoubleArray();
            if (addBias) {
                X[i] = Arrays.copyOfRange(vector, 0, numFeatures + 1); // +1 for the bias column
                X[i][numFeatures] = 1; // setting the bias column
            } else {
                X[i] = Arrays.copyOfRange(vector, 0, numFeatures);

            }
            Y[i] = Arrays.copyOfRange(vector, numFeatures, vector.length);
        }

        // normalizations should be performed in Java because the same transformation is
        // required at prediction time, thus we need to store column means and stds!
        debug("Performing pre-processing");

        switch (normalization) {
        case none:
            // do nothing
            break;
        case zScore:
            means = new double[numFeatures];
            stds = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) { // do not normalize the bias feature!
                for (int j = 0; j < dataNotMissing.numInstances(); j++) { // mean calculation
                    means[i] += X[j][i];
                }
                means[i] /= dataNotMissing.numInstances();
                for (int j = 0; j < dataNotMissing.numInstances(); j++) { // std calculation
                    double diff = X[j][i] - means[i];
                    stds[i] += diff * diff;
                }
                stds[i] = Math.sqrt(stds[i] / dataNotMissing.numInstances());
                for (int j = 0; j < dataNotMissing.numInstances(); j++) { // normalization
                    if (stds[i] != 0) { // check feature not constant
                        X[j][i] = (X[j][i] - means[i]) / stds[i];
                    } else {
                        X[j][i] = 0; // as done in Matlab
                    }
                }
            }
            // System.out.println(Arrays.toString(means));
            // System.out.println(Arrays.toString(stds));
            break;
        case maxFeatureValue:
            // determine maxFeatureValues in one pass and then normalize
            maxFeatureValues = new double[numFeatures];
            Arrays.fill(maxFeatureValues, Double.NEGATIVE_INFINITY);
            for (int j = 0; j < numFeatures; j++) {
                for (int i = 0; i < dataNotMissing.numInstances(); i++) {
                    if (Math.abs(X[i][j]) > maxFeatureValues[j]) {
                        maxFeatureValues[j] = Math.abs(X[i][j]);
                    }
                }
            }
            for (int j = 0; j < numFeatures; j++) {
                for (int i = 0; i < dataNotMissing.numInstances(); i++) {
                    if (maxFeatureValues[j] > 0) {
                        X[i][j] /= maxFeatureValues[j];
                    } else {
                        // do nothing
                    }
                }
            }
            break;
        }
        debug("Pre-processing completed");
        // --- data preparation finished ---
        // ---------------------------------

        // reset the Matlab session
        proxy.eval("clear;"); // clear Matlab's workspace
        proxy.eval("addpath(genpath('" + malsarMatlabPath + "'));"); // add Malsar to Matlab's path
        proxy.eval("rng('default');"); // reset Matlab's random generator
        proxy.eval("maxNumCompThreads(" + maxNumCompThreads + ")");// setting the maximum number of
                                                                   // computational threads

        // send the arrays to MATLAB
        MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
        processor.setNumericArray("X_array", new MatlabNumericArray(X, null));
        processor.setNumericArray("Y_array", new MatlabNumericArray(Y, null));
        // create cell arrays for X and Y
        proxy.eval(xCellArrayMatName + " = cell(1," + numLabels + ");"); // numLabels tasks
        proxy.eval(yCellArrayMatName + " = cell(1," + numLabels + ");"); // numLabels tasks
        for (int i = 0; i < numLabels; i++) {
            // same training set for all tasks
            proxy.eval(xCellArrayMatName + "{" + (i + 1) + "} = X_array;");
            proxy.eval(yCellArrayMatName + "{" + (i + 1) + "} = Y_array(:," + (i + 1) + ");");
        }

        // setting optimization option in Matlab
        proxy.eval("opts = [];");
        proxy.eval("opts.init = " + optInit.getValue() + ";");
        proxy.eval("opts.tFlag = " + optTFlag.getValue() + ";");
        proxy.eval("opts.tol = " + optTol + ";");
        proxy.eval("opts.maxIter = " + optMaxIter + ";");

        // method specific operations start here
        buildInternalSpecific(dataNotMissing);

        // move the learned parameter array to Java
        W = processor.getNumericArray(paramArrayMatName).getRealArray2D();
    }

    /**
     * All Malsar methods should implement their specific behavior in this method.
     * @throws Exception
     */
    protected abstract void buildInternalSpecific(Instances trainingSet) throws Exception;

    @Override
    public TechnicalInformation getTechnicalInformation() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {
        // first pass the instance from the filters
        // nomToBinFilter.input(instance);
        // nomToBinFilter.batchFinished();
        // Instance transformedInstance = nomToBinFilter.output();
        // int numFeatures = transformedInstance.numAttributes() - numLabels;
        int numFeatures = instance.numAttributes() - numLabels;
        // given that the target attributes are not used here, it's ok to replace their values..
        replaceMissingFilter.input(instance);
        replaceMissingFilter.batchFinished();
        Instance transformedInstance = replaceMissingFilter.output();

        // the following assumes that all target attributes are in last positions
        double[] vector = transformedInstance.toDoubleArray();
        double[] X;
        if (addBias) {
            X = Arrays.copyOfRange(vector, 0, numFeatures + 1);
            X[numFeatures] = 1;// the bias terms
        } else {
            X = Arrays.copyOfRange(vector, 0, numFeatures);
        }

        switch (normalization) {
        case none:
            // do nothing
            break;
        case zScore:
            for (int i = 0; i < numFeatures; i++) {
                if (stds[i] != 0) {// feature not constant
                    X[i] = (X[i] - means[i]) / stds[i];
                } else {
                    X[i] = 0;
                }
            }
            break;
        case maxFeatureValue:
            for (int i = 0; i < numFeatures; i++) {
                if (maxFeatureValues[i] > 0) {
                    X[i] /= maxFeatureValues[i];
                }
            }
            break;
        }

        // get the prediction for each target via a simple multiplication with the corresponding
        // column of W
        double[] pValues = new double[numLabels];
        for (int j = 0; j < numLabels; j++) {
            for (int i = 0; i < X.length; i++) {
                pValues[j] += X[i] * W[i][j];
            }
        }
        return new MultiLabelOutput(pValues, true);
    }

    public void setAddBias(boolean addBias) {
        this.addBias = addBias;
    }

    public void setNormalization(normalizationType normalization) {
        this.normalization = normalization;
    }

}
