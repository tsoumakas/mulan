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
 *    ConverterCLUS.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <p>Class that converts a dataset that is originally in the format of the
 * <a href="http://www.cs.kuleuven.be/~dtai/clus/">Clus system</a> to a format
 * that is suitable for Mulan. An arff and an xml file are created.</p>
 * <p>The arff file contains the original dataset with all labels converted to
 * separate attributes and properly converted instances. The xml file contains
 * the hierarchy of the labels.</p>
 *
 * @author George Saridis
 * @author Grigorios Tsoumakas
 * @version 2012.02.06
 */
public class ConverterCLUS {

    /**
     * Converts the original dataset to mulan compatible dataset.
     *
     * @param sourceFilename the source file name
     * @param arffFilename the converted arff name
     * @param xmlFilename the xml name
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public static void convert(String sourceFilename, String arffFilename, String xmlFilename) throws Exception {
        String line;
        try {
            BufferedReader brInput = new BufferedReader(new FileReader(sourceFilename));

            String relationName = null;
            ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
            Instances data = null;
            int numAttributes = 0;
            String[] labelNames = null;
            while ((line = brInput.readLine()) != null) {
                if (line.startsWith("@RELATION")) {
                    relationName = line.replace("@RELATION ", "").replaceAll("'", "").trim();
                    continue;
                }
                if (line.startsWith("@ATTRIBUTE ")) {
                    String tokens[] = line.split("\\s+");
                    Attribute att;
                    if (line.startsWith("@ATTRIBUTE class")) {
                        labelNames = tokens[3].split(",");
                        for (int i = 0; i < labelNames.length; i++) {
                            ArrayList<String> labelValues = new ArrayList<String>();
                            labelValues.add("0");
                            labelValues.add("1");
                            att = new Attribute(labelNames[i], labelValues);
                            attInfo.add(att);
                        }
                    } else {
                        numAttributes++;
                        if (tokens[2].equals("numeric")) {
                            att = new Attribute(tokens[1]);
                        } else {
                            ArrayList<String> nominalValues = new ArrayList<String>();
                            tokens[2].substring(1, tokens[2].length() - 1);
                            String[] nominalTokens = tokens[2].substring(1, tokens[2].length() - 1).split(",");
                            nominalValues.addAll(Arrays.asList(nominalTokens));
                            att = new Attribute(tokens[1], nominalValues);
                        }
                        attInfo.add(att);
                    }
                    continue;
                }
                if (line.toLowerCase().startsWith("@data")) {
                    data = new Instances(relationName, attInfo, 0);
                    while ((line = brInput.readLine()) != null) {
                        // fill data
                        String[] tokens = line.split(",");
                        double[] values = new double[attInfo.size()];
                        for (int i = 0; i < numAttributes; i++) {
                            Attribute att = (Attribute) attInfo.get(i);
                            if (att.isNumeric()) {
                                values[i] = Double.parseDouble(tokens[i]);
                            } else {
                                values[i] = att.indexOfValue(tokens[i]);
                            }
                        }
                        String[] labels = tokens[numAttributes].split("@");
                        // fill class values
                        for (int j = 0; j < labels.length; j++) {
                            String[] splitedLabels = labels[j].split("/");
                            String attrName = splitedLabels[0];
                            Attribute att = data.attribute(attrName);
                            values[attInfo.indexOf(att)] = 1;
                            for (int k = 1; k < splitedLabels.length; k++) {
                                attrName = attrName + "/" + splitedLabels[k];
                                att = data.attribute(attrName);
                                values[attInfo.indexOf(att)] = 1;
                            }
                        }
                        Instance instance = new DenseInstance(1, values);
                        data.add(instance);
                    }
                }
            }
            BufferedWriter writer;
            writer = new BufferedWriter(new FileWriter(arffFilename));
            writer.write(data.toString());
            writer.close();

            // write xml file
            writer = new BufferedWriter(new FileWriter(xmlFilename));
            writer.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
            writer.write("<labels xmlns=\"http://mulan.sourceforge.net/labels\">\n");
            writer.write("<label name=\"" + labelNames[0] + "\">");
            int depth = 0;
            for (int i = 1; i < labelNames.length; i++) {
                int difSlashes = countSlashes(labelNames[i]) - countSlashes(labelNames[i - 1]);
                // child
                if (difSlashes == 1) {
                    depth++;
                    writer.write("\n");
                    for (int j = 0; j < depth; j++) {
                        writer.write("\t");
                    }
                    writer.write("<label name=\"" + labelNames[i] + "\">");
                }
                // sibling
                if (difSlashes == 0) {
                    writer.write("</label>\n");
                    for (int j = 0; j < depth; j++) {
                        writer.write("\t");
                    }
                    writer.write("<label name=\"" + labelNames[i] + "\">");
                }
                // ancestor
                if (difSlashes < 0) {
                    writer.write("</label>\n");
                    for (int j = 0; j < Math.abs(difSlashes); j++) {
                        depth--;
                        for (int k = 0; k < depth; k++) {
                            writer.write("\t");
                        }
                        writer.write("</label>\n");
                    }
                    for (int j = 0; j < depth; j++) {
                        writer.write("\t");
                    }
                    writer.write("<label name=\"" + labelNames[i] + "\">");
                }
            }
            writer.write("</label>\n");
            while (depth > 0) {
                for (int k = 0; k < depth; k++) {
                    writer.write("\t");
                }
                writer.write("</label>\n");
                depth--;
            }
            writer.write("</labels>");
            writer.close();


        } catch (IOException ioEx) {
            ioEx.printStackTrace();
        }
    }

    private static int countSlashes(String label) {
        int counter = 0;
        for (int i = 0; i < label.length(); i++) {
            if (label.charAt(i) == '/') {
                counter++;
            }
        }
        return counter;
    }
}