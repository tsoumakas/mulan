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
 *    MultipleEvaluation.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.evaluation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import mulan.core.MulanException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.MacroAverageMeasure;
import mulan.evaluation.measure.Measure;

/**
 * Simple class that includes an array, whose elements are lists of evaluation evaluations. Used to
 * compute means and standard deviations of multiple evaluations (e.g. cross-validation)
 * 
 * @author Grigorios Tsoumakas
 */
public class MultipleEvaluation {

    private MultiLabelInstances data;
    private ArrayList<Evaluation> evaluations;
    private HashMap<String, Double> mean;
    private HashMap<String, Double> standardDeviation;
    private HashMap<String, Double[]> labelMean;
    private HashMap<String, Double[]> labelStandardDeviation;

    /**
     * Constructs a new object
     * 
     * @param data the evaluation data used for obtaining label names for per outputting per label
     *            values of macro average measures
     */
    public MultipleEvaluation(MultiLabelInstances data) {
        evaluations = new ArrayList<Evaluation>();
        this.data = data;
    }

    /**
     * Constructs a new object with given array of evaluations
     * 
     * @param data the evaluation data used for obtaining label names for per outputting per label
     *            values of macro average measures
     * @param someEvaluations the array of evaluations
     */
    public MultipleEvaluation(Evaluation[] someEvaluations, MultiLabelInstances data) {
        evaluations = new ArrayList<Evaluation>();
        evaluations.addAll(Arrays.asList(someEvaluations));
        this.data = data;
    }

    /**
     * Computes mean and standard deviation of all evaluation measures
     */
    public void calculateStatistics() {
        int size = evaluations.size();
        HashMap<String, Double> sums = new HashMap<String, Double>();
        HashMap<String, Double[]> labelSums = new HashMap<String, Double[]>();

        // calculate sums of measures
        for (int i = 0; i < evaluations.size(); i++) {
            for (Measure m : evaluations.get(i).getMeasures()) {
                double value = Double.NaN;
                try {
                    value = m.getValue();
                } catch (Exception ex) {
                }
                if (sums.containsKey(m.getName())) {
                    sums.put(m.getName(), sums.get(m.getName()) + value);
                } else {
                    sums.put(m.getName(), value);
                }
                if (m instanceof MacroAverageMeasure) {
                    Double[] v = new Double[data.getNumLabels()];
                    for (int j = 0; j < data.getNumLabels(); j++)
                        v[j] = ((MacroAverageMeasure) m).getValue(j);
                    if (labelSums.containsKey(m.getName())) {
                        Double[] v_old = labelSums.get(m.getName());
                        for (int j = 0; j < data.getNumLabels(); j++)
                            v[j] += v_old[j];
                    }
                    labelSums.put(m.getName(), v);
                }
            }
        }
        mean = new HashMap<String, Double>();
        for (String measureName : sums.keySet()) {
            mean.put(measureName, sums.get(measureName) / size);
        }

        labelMean = new HashMap<String, Double[]>();
        for (String measureName : labelSums.keySet()) {
            Double[] v = labelSums.get(measureName);
            for (int j = 0; j < data.getNumLabels(); j++)
                v[j] = v[j] / size;
            labelMean.put(measureName, v);
        }

        // calculate sums of squared differences from mean
        sums = new HashMap<String, Double>();
        labelSums = new HashMap<String, Double[]>();

        for (int i = 0; i < evaluations.size(); i++) {
            for (Measure m : evaluations.get(i).getMeasures()) {
                double value = Double.NaN;
                try {
                    value = m.getValue();
                } catch (Exception ex) {
                }
                if (sums.containsKey(m.getName())) {
                    sums.put(m.getName(),
                            sums.get(m.getName()) + Math.pow(value - mean.get(m.getName()), 2));
                } else {
                    sums.put(m.getName(), Math.pow(value - mean.get(m.getName()), 2));
                }
                if (m instanceof MacroAverageMeasure) {
                    Double[] mean = labelMean.get(m.getName());
                    Double[] v = new Double[data.getNumLabels()];
                    for (int j = 0; j < data.getNumLabels(); j++)
                        v[j] = ((MacroAverageMeasure) m).getValue(j);
                    if (labelSums.containsKey(m.getName())) {
                        Double[] v_old = labelSums.get(m.getName());
                        for (int j = 0; j < data.getNumLabels(); j++)
                            v[j] = Math.pow(v[j] - mean[j], 2) + v_old[j];
                    }
                    labelSums.put(m.getName(), v);
                }
            }
        }
        standardDeviation = new HashMap<String, Double>();
        for (String measureName : sums.keySet()) {
            standardDeviation.put(measureName, Math.sqrt(sums.get(measureName) / size));
        }
        labelStandardDeviation = new HashMap<String, Double[]>();
        for (String measureName : labelSums.keySet()) {
            Double[] s = labelSums.get(measureName);
            for (int j = 0; j < data.getNumLabels(); j++) {
                s[j] /= size;
            }
            labelStandardDeviation.put(measureName, s);
        }
    }

    /**
     * Adds an evaluation results to the list of evaluations
     * 
     * @param evaluation an evaluation result
     */
    public void addEvaluation(Evaluation evaluation) {
        evaluations.add(evaluation);
    }

    /**
     * Returns a string with the results of the evaluation
     * 
     * @return a string with the results of the evaluation
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Measure m : evaluations.get(0).getMeasures()) {
            String measureName = m.getName();
            sb.append(measureName);
            sb.append(": ");
            sb.append(String.format("%.4f", mean.get(measureName)));
            sb.append("\u00B1");
            sb.append(String.format("%.4f", standardDeviation.get(measureName)));
            sb.append("\n");
            if (m instanceof MacroAverageMeasure) {
                Double[] v = labelMean.get(measureName);
                Double[] s = labelStandardDeviation.get(measureName);
                for (int i = 0; i < data.getNumLabels(); i++) {
                    sb.append(data.getDataSet().attribute(data.getLabelIndices()[i]).name())
                            .append(": ");
                    sb.append(String.format("%.4f", v[i]));
                    sb.append("\u00B1");
                    sb.append(String.format("%.4f", s[i]));
                    sb.append(" ");
                }
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Returns the mean value of a measure
     * 
     * @param measureName the name of the measure
     * @return the mean value of the measure
     */
    public double getMean(String measureName) {
        return mean.get(measureName);
    }

    /**
     * Returns the mean value of a specific label of a macro-averaged measure
     * 
     * @param measureName the name of the measure
     * @param labelIndex the label index
     * @return the mean value of the measure for the given label index
     * @throws MulanException when the measure is not macro-averaged
     */
    public double getMean(String measureName, int labelIndex) throws MulanException {
        if (!labelMean.containsKey(measureName)) {
            throw new MulanException("Not a macro-averaged measure!");
        }
        return labelMean.get(measureName)[labelIndex];
    }

    /**
     * Returns the standard deviation of a measure
     * 
     * @param measureName the name of the measure
     * @return the standard deviation of the measure
     */
    public double getStd(String measureName) {
        return standardDeviation.get(measureName);
    }

    /**
     * Returns the standard deviation of a specific label of a macro-averaged measure
     * 
     * @param measureName the name of the measure
     * @param labelIndex the label index
     * @return the standard deviation of the measure for the given label index
     * @throws MulanException when the measure is not macro-averaged
     */
    public double getStd(String measureName, int labelIndex) throws MulanException {
        if (!labelStandardDeviation.containsKey(measureName)) {
            throw new MulanException("Not a macro-averaged measure!");
        }
        return labelStandardDeviation.get(measureName)[labelIndex];
    }

    /**
     * Returns a CSV string representation of the results
     * 
     * @return a CSV string representation of the results
     */
    public String toCSV() {
        StringBuilder sb = new StringBuilder();
        for (Measure m : evaluations.get(0).getMeasures()) {
            String measureName = m.getName();
            sb.append(String.format("%.4f", mean.get(measureName)));
            sb.append("\u00B1");
            sb.append(String.format("%.4f", standardDeviation.get(measureName)));
            sb.append(";");
        }
        return sb.toString();
    }

    public ArrayList<Evaluation> getEvaluations() {
        return evaluations;
    }
}