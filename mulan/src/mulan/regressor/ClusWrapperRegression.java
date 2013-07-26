package mulan.regressor;

import java.io.IOException;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.clus.ClusWrapperClassification;
import weka.core.Instance;

/**
 * This class implements a wrapper for the multi-target regression methods included in the CLUS
 * library.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 11.27.2012
 * 
 */
public class ClusWrapperRegression extends ClusWrapperClassification {

    private static final long serialVersionUID = 1L;

    public ClusWrapperRegression(String clusWorkingDir, String datasetName, String settingsFilePath)
            throws IOException {
        super(clusWorkingDir, datasetName, settingsFilePath);
    }

    /**
     * This method exists so that CLUSWrapperRegression can extend MultiLabelLearnerBase. Also helps the
     * Evaluator to determine the type of the MultiLabelOutput and thus prepare the appropriate
     * measures to be evaluated upon.
     */
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {
        double[] pValues = new double[numLabels];
        return new MultiLabelOutput(pValues, true);

    }
}
