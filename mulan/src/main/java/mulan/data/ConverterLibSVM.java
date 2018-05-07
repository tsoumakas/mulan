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

/*
 *    ConverterLibSVM.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;
import weka.core.*;

/**
 * Class that converts LibSVM multi-label data sets to Mulan compatible format <p>
 *
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.01 $
 */
public class ConverterLibSVM {

    /**
     * Converts a multi-label dataset from LibSVM format to the format
     * that is compatible with Mulan. It constructs one ARFF and one XML file. 
     *
     * @param path the directory that contains the source file and will contain 
     * the target files
     * @param sourceFilename the name of the source file
     * @param relationName the relation name of the arff file that will be 
     * constructed
     * @param targetFilestem the filestem for the target files (.arff and .xml)
     */
    public static void convertFromLibSVM(String path, String sourceFilename, String targetFilestem, String relationName) {
        BufferedReader aReader = null;
        BufferedWriter aWriter = null;

        int numLabels = 0;
        int numAttributes = 0;
        int numInstances = 0;
        double meanParsedAttributes = 0;

        // Calculate number of labels and attributes

        String Line = null;
        try {
            aReader = new BufferedReader(new FileReader(path + sourceFilename));

            while ((Line = aReader.readLine()) != null) {
                numInstances++;

                StringTokenizer strTok = new StringTokenizer(Line, " ");
                while (strTok.hasMoreTokens()) {
                    String token = strTok.nextToken();

                    if (token.indexOf(":") == -1) {
                        // parse label info
                        StringTokenizer labelTok = new StringTokenizer(token, ",");
                        while (labelTok.hasMoreTokens()) {
                            String strLabel = labelTok.nextToken();
                            int intLabel = Integer.parseInt(strLabel);
                            if (intLabel > numLabels) {
                                numLabels = intLabel;
                            }
                        }
                    } else {
                        // parse attribute info
                        meanParsedAttributes++;
                        StringTokenizer attrTok = new StringTokenizer(token, ":");
                        String strAttrIndex = attrTok.nextToken();
                        int intAttrIndex = Integer.parseInt(strAttrIndex);
                        if (intAttrIndex > numAttributes) {
                            numAttributes = intAttrIndex;
                        }
                    }
                }
            }

            numLabels++;

            System.out.println("Number of attributes: " + numAttributes);
            System.out.println("Number of instances: " + numInstances);
            System.out.println("Number of classes: " + numLabels);

            System.out.println("Constructing XML file... ");
            LabelsMetaDataImpl meta = new LabelsMetaDataImpl();
            for (int label = 0; label < numLabels; label++) {
                meta.addRootNode(new LabelNodeImpl("Label" + (label + 1)));
            }

            String labelsFilePath = path + targetFilestem + ".xml";
            try {
                LabelsBuilder.dumpLabels(meta, labelsFilePath);
                System.out.println("Done!");
            } catch (LabelsBuilderException e) {
                File labelsFile = new File(labelsFilePath);
                if (labelsFile.exists()) {
                    labelsFile.delete();
                }
                System.out.println("Construction of labels XML failed!");
            }

            meanParsedAttributes /= numInstances;
            boolean Sparse = false;
            if (meanParsedAttributes < numAttributes) {
                Sparse = true;
                System.out.println("Dataset is sparse.");
            }

            // Define Instances class to hold data
            ArrayList<Attribute> attInfo = new ArrayList<Attribute>(numAttributes + numLabels);
            Attribute[] att = new Attribute[numAttributes + numLabels];

            for (int i = 0; i < numAttributes; i++) {
                att[i] = new Attribute("Att" + (i + 1));
                attInfo.add(att[i]);
            }
            ArrayList<String> ClassValues = new ArrayList<String>(2);
            ClassValues.add("0");
            ClassValues.add("1");
            for (int i = 0; i < numLabels; i++) {
                att[numAttributes + i] = new Attribute("Label" + (i + 1), ClassValues);
                attInfo.add(att[numAttributes + i]);
            }

            // Re-read file and convert into multi-label arff
            int countInstances = 0;

            aWriter = new BufferedWriter(new FileWriter(path + targetFilestem + ".arff"));
            Instances data = new Instances(relationName, attInfo, 0);
            aWriter.write(data.toString());

            aReader = new BufferedReader(new FileReader(path + sourceFilename));

            while ((Line = aReader.readLine()) != null) {
                countInstances++;

                // set all  values to 0
                double[] attValues = new double[numAttributes + numLabels];
                Arrays.fill(attValues, 0);

                Instance tempInstance = new DenseInstance(1, attValues);
                tempInstance.setDataset(data);

                // separate class info from attribute info
                // ensure class info exists
                StringTokenizer strTok = new StringTokenizer(Line, " ");

                while (strTok.hasMoreTokens()) {
                    String token = strTok.nextToken();

                    if (token.indexOf(":") == -1) {
                        // parse label info
                        StringTokenizer labelTok = new StringTokenizer(token, ",");
                        while (labelTok.hasMoreTokens()) {
                            String strLabel = labelTok.nextToken();
                            int intLabel = Integer.parseInt(strLabel);
                            tempInstance.setValue(numAttributes + intLabel, 1);
                        }
                    } else {
                        // parse attribute info
                        StringTokenizer AttrTok = new StringTokenizer(token, ":");
                        String strAttrIndex = AttrTok.nextToken();
                        String strAttrValue = AttrTok.nextToken();
                        tempInstance.setValue(Integer.parseInt(strAttrIndex) - 1, Double.parseDouble(strAttrValue));
                    }
                }

                if (Sparse) {
                    SparseInstance tempSparseInstance = new SparseInstance(tempInstance);
                    aWriter.write(tempSparseInstance.toString() + "\n");
                } else {
                    aWriter.write(tempInstance.toString() + "\n");
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (aReader != null) {
                    aReader.close();
                }
                if (aWriter != null) {
                    aWriter.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    private static void createLabelsMetadataFile(String filePath, int numLabels) throws LabelsBuilderException {
    }

    /**
     * Command line interface for the converter
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        String path = null;
        String source = null;
        String target = null;
        String relationName = "LibSVM";
        try {
            path = Utils.getOption("path", args);
            source = Utils.getOption("source", args);
            target = Utils.getOption("target", args);
            relationName = Utils.getOption("name", args);
            ConverterLibSVM.convertFromLibSVM(path, source, target, relationName);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}