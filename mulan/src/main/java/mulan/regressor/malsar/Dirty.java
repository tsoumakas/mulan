package mulan.regressor.malsar;

import java.util.Arrays;

import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabTypeConverter;
import weka.core.Instances;

/**
 * TODO Add publication info for this method!
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class Dirty extends MalsarWrapper{

    /**
     * Number of internal cv folds used for parameter optimization
     */
    protected int numCVFolds = 5;

    /**
     * The values of the constant c used in computation of regularization weights lambda_b and
     * lambda_s. The following values were selected based on Jalali et al.
     */
    protected double[] c = { 0.01, 0.1, 1, 10, 100 }; // Jalali

    protected int numScalingSteps = 5;

    public Dirty(String malsarPath, MatlabProxy proxy, int maxNumCompThreads) {
        super(malsarPath, proxy, maxNumCompThreads);
        // setting method specific optimization options
        optInit = optsInit.zero; // Malsar's example
        optTFlag = optsTFlag.one; // Malsar's example
        optTol = 0.00001; // Aho et al. states E10^-10
        // much larger than Malsar's default (negligible differences when turning from 1000 to 5000)
        optMaxIter = 5000;
        // setting feature normalization options
        addBias = false; // nothing stated in Malsar or Jalali et al.
        normalization = normalizationType.maxFeatureValue; // from Jalali et al.
    }

    @Override
    protected void buildInternalSpecific(Instances trainingSet) throws Exception {
        // compute the ranges of the lambda_b and lambda_s parameters of dirty based on
        // characteristics of the dataset and the different values of the c constant.
        // the passed Instances object is already transformed
        int numFeatures = trainingSet.numAttributes() - numLabels;
        int numExamples = trainingSet.numInstances();
        double[] lambda_b = new double[c.length];
        double[][] lambda_s = new double[c.length][];
        double problemCapacity = Math.sqrt(numLabels * Math.log10(numFeatures) / numExamples);
        for (int i = 0; i < c.length; i++) {
            lambda_b[i] = c[i] * problemCapacity;
            // for each lambda_b[i], select lambda_s[i]'s such that 1/numLabels <=
            // lambda_s[i]/lambda_b[i] <= 1
            lambda_s[i] = new double[numScalingSteps];
            double scalingStep = (double) (numLabels - 1) / (numScalingSteps - 1);
            for (int j = 0; j < numScalingSteps; j++) {
                // System.out.println("Dividing by " + (1 + (j * scalingStep)));
                lambda_s[i][j] = lambda_b[i] / (1 + (j * scalingStep));
                double ratio = lambda_s[i][j] / lambda_b[i];
                if (ratio > 1 || ratio < 1 / numLabels) {
                    throw new Exception("Unexpected");
                }
            }
        }

        debug("Performing model selection via cross validation..");
        double bestPerformance = Double.MAX_VALUE;
        int bestLambdaBIndex = -1;
        int bestLambdaSIndex = -1;
        for (int j = 0; j < lambda_b.length; j++) {
            proxy.eval("lambda_b = " + lambda_b[j] + ";");
            proxy.eval("lambda_s = " + Arrays.toString(lambda_s[j]).replace(",", ";") + ";");

            proxy.eval("[best_param perform_mat] = CrossValidation2Param (" + xCellArrayMatName
                    + ", " + yCellArrayMatName + ", 'Least_Dirty', " + optsMatName
                    + ", lambda_b, lambda_s, " + numCVFolds + ", 'eval_MTL_mse', false);");
            MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
            double[][] performances = processor.getNumericArray("perform_mat").getRealArray2D();
            for (int i = 0; i < lambda_s[j].length; i++) {
                debug("lambda_b: " + lambda_b[j] + "," + "lambda_s: " + lambda_s[j][i] + "\t"
                        + performances[i][0]);
                if (performances[i][0] < bestPerformance) { // lower is better
                    bestPerformance = performances[i][0];
                    bestLambdaBIndex = j;
                    bestLambdaSIndex = i;
                }
            }
        }
        debug("..completed, best parameter set: (" + lambda_b[bestLambdaBIndex] + ","
                + lambda_s[bestLambdaBIndex][bestLambdaSIndex] + ")");

        debug("Builing final model..");
        proxy.eval(paramArrayMatName + " = Least_Dirty(" + xCellArrayMatName + ", "
                + yCellArrayMatName + ", " + lambda_b[bestLambdaBIndex] + ", "
                + lambda_s[bestLambdaBIndex][bestLambdaSIndex] + ", " + optsMatName + ");");
        debug("..completed");

    }

    /**
     * This classifier implements its own makeCopy method because the MatlabProxy object is not
     * Serializable
     */
    public Dirty makeCopy() throws Exception {
        Dirty malsar = new Dirty(malsarMatlabPath, proxy, maxNumCompThreads);
        malsar.setDebug(this.getDebug());
        malsar.setNormalization(normalization);
        malsar.setAddBias(addBias);
        return malsar;
    }

    public void setC(double[] c) {
        this.c = c;
    }

    public void setNumCVFolds(int numCVFolds) {
        this.numCVFolds = numCVFolds;
    }

    public void setNumScalingSteps(int numScalingSteps) {
        this.numScalingSteps = numScalingSteps;
    }

}
