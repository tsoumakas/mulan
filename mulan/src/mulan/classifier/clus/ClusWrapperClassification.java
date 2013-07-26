/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.clus;

import java.io.*;
import java.nio.channels.FileChannel;

import mulan.classifier.*;
import mulan.data.MultiLabelInstances;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.SparseToNonSparse;

/**
 * This class implements a wrapper for the multi-label classification methods included in the CLUS
 * library.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 07.26.2013
 * 
 */
public class ClusWrapperClassification extends MultiLabelLearnerBase {

    /**
     * The path to a directory where all the temporary files needed by the CLUS library are written.
     */
    private String clusWorkingDir;
    /**
     * The dataset name that will be used for training, test and settings files.
     */
    private String datasetName;

    /**
     * Path to the settings file.
     */
    private String settingsFilePath;

    /**
     * Whether an ensemble method will be used.
     */
    private boolean isEnsemble = false;

    /**
     * Whether a rule-based method will be used.
     */
    private boolean isRuleBased = false;

    /**
     * Constructor.
     * 
     * @param clusWorkingDir
     * @param datasetName
     * @param settingsFilePath
     * @throws IOException
     */
    public ClusWrapperClassification(String clusWorkingDir, String datasetName,
            String settingsFilePath) throws IOException {
        this.clusWorkingDir = clusWorkingDir;
        this.datasetName = datasetName;
        this.settingsFilePath = settingsFilePath;
    }

    /**
     * This method does the following:
     * <ol>
     * <li>Creates a working directory for the CLUS library</li>
     * <li>Copies the supplied settings file in this directory</li>
     * <li>Modifies the File, TestSet and Target lines of the settings file to the appropriate
     * values</li>
     * <li>Makes the supplied training set CLUS compliant and copies it to the working directory</li>
     * </ol>
     */
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        // create the CLUS working directory if it does not exist
        File theDir = new File(clusWorkingDir);
        if (!theDir.exists()) {
            System.out.println("Creating CLUS working directory: " + clusWorkingDir);
            boolean result = theDir.mkdir();
            if (result) {
                System.out.println("CLUS working directory created");
            }
        }

        // copy the settings file in the working directory
        File settingsFile = new File(settingsFilePath);
        File newSettingsFile = new File(clusWorkingDir + this.datasetName + "-train.s");
        copyFile(settingsFile, newSettingsFile);

        // modify the File, TestSet and Target lines of the settings file to the appropriate values
        BufferedReader in = new BufferedReader(new FileReader(newSettingsFile));
        String settings = "";
        String line;
        while ((line = in.readLine()) != null) {
            if (line.startsWith("File")) {
                settings += "File = " + clusWorkingDir + this.datasetName + "-train.arff" + "\n";
                continue;
            }
            if (line.startsWith("TestSet")) {
                settings += "TestSet = " + clusWorkingDir + this.datasetName + "-test.arff" + "\n";
                continue;
            }
            if (line.startsWith("Target")) {
                settings += "Target = ";
                for (int i = 0; i < numLabels - 1; i++) {
                    // all targets except last
                    settings += (labelIndices[i] + 1) + ",";
                }
                // last target
                settings += (labelIndices[numLabels - 1] + 1) + "\n";
                continue;
            }
            settings += line + "\n";
        }
        in.close();

        BufferedWriter out = new BufferedWriter(new FileWriter(newSettingsFile));
        out.write(settings);
        out.close();

        // transform the supplied MultilabelInstances object in an arff formated file (accepted by
        // CLUS) and write the file in the working directory with the appropriate name
        makeClusCompliant(trainingSet, clusWorkingDir + datasetName + "-train.arff");

    }

    /**
     * This method exists so that CLUSWrapperClassification can extend MultiLabelLearnerBase. Also helps the
     * Evaluator to determine the type of the MultiLabelOutput and thus prepare the appropriate
     * measures to be evaluated upon.
     */
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {
        double[] confidences = new double[numLabels];
        return new MultiLabelOutput(confidences, 0.5);

    }

    public String getClusWorkingDir() {
        return clusWorkingDir;
    }

    public String getDatasetName() {
        return datasetName;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        // TODO Add Technical Information!
        return null;
    }

    public boolean isEnsemble() {
        return isEnsemble;
    }

    public void setEnsemble(boolean isEnsemble) {
        this.isEnsemble = isEnsemble;
    }

    public boolean isRuleBased() {
        return isRuleBased;
    }

    public void setRuleBased(boolean isRuleBased) {
        this.isRuleBased = isRuleBased;
    }

    private static final long serialVersionUID = 1L;

    private static void copyFile(File sourceFile, File destFile) throws IOException {
        if (!destFile.exists()) {
            destFile.createNewFile();
        }

        FileChannel source = null;
        FileChannel destination = null;

        try {
            source = new FileInputStream(sourceFile).getChannel();
            destination = new FileOutputStream(destFile).getChannel();
            destination.transferFrom(source, 0, source.size());
        } finally {
            if (source != null) {
                source.close();
            }
            if (destination != null) {
                destination.close();
            }
        }
    }

    /**
     * Takes a dataset as a MultiLabelInstances object and writes an arff file which is CLUS
     * compliant.
     * 
     * @param mlDataset
     * @param fileName
     * @throws Exception
     */
    public static void makeClusCompliant(MultiLabelInstances mlDataset, String fileName)
            throws Exception {
        BufferedWriter out = new BufferedWriter(new FileWriter(new File(fileName)));

        // the file will be written in the datasetPath directory
        Instances dataset = mlDataset.getDataSet();
        SparseToNonSparse stns = new SparseToNonSparse(); // new instance of filter
        stns.setInputFormat(dataset); // inform filter about dataset **AFTER** setting options
        Instances nonSparseDataset = Filter.useFilter(dataset, stns); // apply filter

        String header = new Instances(nonSparseDataset, 0).toString();
        // preprocess the header
        // remove ; characters and truncate long attribute names
        String[] headerLines = header.split("\n");
        for (int i = 0; i < headerLines.length; i++) {
            if (headerLines[i].startsWith("@attribute")) {
                headerLines[i] = headerLines[i].replaceAll(";", "SEMI_COLON");
                String originalAttributeName = headerLines[i].split(" ")[1];
                String newAttributeName = originalAttributeName;
                if (originalAttributeName.length() > 30) {
                    newAttributeName = originalAttributeName.substring(0, 30) + "..";
                }
                out.write(headerLines[i].replace(originalAttributeName, newAttributeName) + "\n");
            } else {
                out.write(headerLines[i] + "\n");
            }
        }
        for (int i = 0; i < nonSparseDataset.numInstances(); i++) {
            if (i % 100 == 0) {
                out.flush();
            }
            out.write(nonSparseDataset.instance(i) + "\n");
        }
        out.close();
    }
}