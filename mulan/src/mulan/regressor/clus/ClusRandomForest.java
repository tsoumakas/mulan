package mulan.regressor.clus;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import mulan.data.MultiLabelInstances;

/**
 * This class is a wrapper for the Multi-Target Random Forest (MORF) algorithm implemented in <a
 * href="https://dtai.cs.kuleuven.be/clus/">CLUS</a> library.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.04.01
 * 
 */
public class ClusRandomForest extends ClusWrapperRegression {

    private static final long serialVersionUID = 1L;
    /**
     * The number of random trees in the ensemble.
     */
    private int numTrees;

    /**
     * Constructor.
     * 
     * @param clusWorkingDir the working directory
     * @param datasetName the dataset name
     * @param numTrees the number of trees
     */
    public ClusRandomForest(String clusWorkingDir, String datasetName, int numTrees) {
        super(clusWorkingDir, datasetName);
        this.isEnsemble = true;
        this.numTrees = numTrees;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        super.buildInternal(trainingSet);
        createSettingsFile();
    }

    /**
     * This method creates a CLUS settings file that corresponds to the MORF algorithm and writes it in
     * clusWorkingDir.
     * 
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void createSettingsFile() throws Exception {
        BufferedWriter out = new BufferedWriter(new FileWriter(new File(clusWorkingDir + this.datasetName
                + "-train.s")));
        out.write("[General]\nVerbose = 0\n\n[Data]\n");
        out.write("File = " + clusWorkingDir + this.datasetName + "-train.arff" + "\n");
        out.write("TestSet = " + clusWorkingDir + this.datasetName + "-test.arff" + "\n");
        out.write("\n[Attributes]\n");
        out.write("Target = ");
        for (int i = 0; i < numLabels - 1; i++) {// all targets except last
            out.write((labelIndices[i] + 1) + ",");
        }
        out.write((labelIndices[numLabels - 1] + 1) + "\n\n"); // last target
        out.write("[Ensemble]\nIterations = " + numTrees + "\n");
        out.write("EnsembleMethod = RForest\n\n[Output]\nWritePredictions = Test\n");
        out.close();
    }
}
