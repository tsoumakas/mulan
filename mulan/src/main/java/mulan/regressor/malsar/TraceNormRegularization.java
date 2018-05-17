package mulan.regressor.malsar;

import java.util.Arrays;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabTypeConverter;
import weka.core.Instances;

/**
 * TODO Add publication info for this method!
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class TraceNormRegularization extends MalsarWrapper{

    /** The number of internal cv folds used for parameter optimization (from Malsar). */
    protected int numCVFolds = 5;
    /**
     * The values of rho (rho controls the rank of W) that will be tested via internal cv (from
     * Malsar).
     */
    protected double[] rho = { 0.001, 0.01, 0.1, 1, 10, 100, 1000 };

    public TraceNormRegularization(String malsarPath, MatlabProxy proxy, int maxNumCompThreads) {
        super(malsarPath, proxy, maxNumCompThreads);
        // setting method specific optimization options (same as in Malsar's example)
        optInit = optsInit.zero;
        optTFlag = optsTFlag.one;
        optTol = 0.00001;
        optMaxIter = 5000;
        // setting feature normalization options (same as in Malsar's example)
        addBias = true;
        normalization = normalizationType.zScore;
    }

    /**
     * The passed MultiLabelInstances object is not used in this case.
     */
    @Override
    protected void buildInternalSpecific(Instances trainingSet) throws MatlabInvocationException {
        // setting variables related to parameter (model) selection
        proxy.eval("eval_func_str = 'eval_MTL_mse';"); // evaluation function
        proxy.eval("higher_better = false;"); // the lower the mse the better
        proxy.eval("cv_fold = " + numCVFolds + ";"); // number of cv folds used for parameter tuning
        proxy.eval("rho_1 = " + Arrays.toString(rho).replace(",", ";") + ";");
        debug("Performing model selection via cross validation..");
        proxy.eval("[best_param perform_mat] = CrossValidation1Param(" + xCellArrayMatName + ", "
                + yCellArrayMatName + ", 'Least_Trace', " + optsMatName
                + ", rho_1, cv_fold, eval_func_str, higher_better);");
        MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
        double[][] performances = processor.getNumericArray("perform_mat").getRealArray2D();
        for (int i = 0; i < rho.length; i++) { // print the performance for each parameter
            debug("Param: " + rho[i] + "\t" + performances[i][0]);
        }
        double bestParam = ((double[]) proxy.getVariable("best_param"))[0];
        debug("..completed, best parameter: " + bestParam);

        debug("Builing final model..");
        proxy.eval(paramArrayMatName + " = Least_Trace(" + xCellArrayMatName + ", "
                + yCellArrayMatName + ", best_param, " + optsMatName + ");");
        debug("..completed");
    }

    /**
     * This classifier implements its own makeCopy method because the MatlabProxy object is not
     * Serializable.
     */
    public TraceNormRegularization makeCopy() throws Exception {
        TraceNormRegularization malsar = new TraceNormRegularization(malsarMatlabPath, proxy,
                maxNumCompThreads);
        malsar.setDebug(this.getDebug());
        return malsar;
    }

    public void setNumCVFolds(int numCVFolds) {
        this.numCVFolds = numCVFolds;
    }

    public void setRho(double[] rho) {
        this.rho = rho;
    }

}
