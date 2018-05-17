package mulan.regressor.clus;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.clus.ClusWrapperClassification;
import weka.core.Instance;

/**
 * This class is a wrapper for the multi-target regression methods included in 
 * <a href="https://dtai.cs.kuleuven.be/clus/">CLUS</a> library.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.04.01
 */
public class ClusWrapperRegression extends ClusWrapperClassification {

    private static final long serialVersionUID = 1L;

    public ClusWrapperRegression(String clusWorkingDir, String datasetName) {
        super(clusWorkingDir, datasetName);
    }

    public ClusWrapperRegression(String clusWorkingDir, String datasetName, String settingsFilePath) {
        super(clusWorkingDir, datasetName, settingsFilePath);
    }

    /**
     * This method exists so that CLUSWrapperRegression can extend MultiLabelLearnerBase. Also helps the
     * Evaluator to determine the type of the MultiLabelOutput and thus prepare the appropriate measures.
     */
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {
        double[] pValues = new double[numLabels];
        return new MultiLabelOutput(pValues, true);

    }
}
