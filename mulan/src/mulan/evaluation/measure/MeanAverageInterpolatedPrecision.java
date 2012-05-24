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
 *    MeanAverageInterpolatedPrecision.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import java.util.Collections;

import weka.core.Utils;

/**
 * Implementation of MAiP (Mean Average Interpolated Precision)
 *
 * @author Fragkiskos Chatziasimidis
 * @author John Panagos
 * @version 2012.05.24
 */
public class MeanAverageInterpolatedPrecision extends LabelBasedAveragePrecision {

    private int recallLevels;
    private double[][] precision;//gia kathe label to precision gia kathe record tou test arff 
    private double[][] recall;//gia kathe label to recall gia kathe record tou test arff
    private double[][] interpolatedPrecision;//gia kathe label to interpolated precision gia kathe recall level
    protected double[] averageInterpolatedPrecision;//gia kathe label o mesos oros twn interpolated
    private double[] recallLevel;//pinakas me ta recall levels edw einai deka 

    /**
     * Constructor
     *
     * @param numOfLabels the number of labels
     * @param recallLevels the number of standard recall levels uniformly
     * distributed in [0,1]
     */
    public MeanAverageInterpolatedPrecision(int numOfLabels, int recallLevels) {
        super(numOfLabels);
        this.recallLevels = recallLevels;
    }

    @Override
    public String getName() {
        return "Mean Average Interpolated Precision";
    }

    protected void calculateAverageInterpolatedPrecisions() {
        recallLevel = new double[recallLevels];
        for (int j = 0; j < recallLevels; j++) {
            recallLevel[j] = (double) j / (recallLevels - 1);
        }
        //arxikopoiw tous pinakes
        averageInterpolatedPrecision = new double[numOfLabels];
        precision = new double[confact[1].size()][numOfLabels];
        recall = new double[confact[1].size()][numOfLabels];
        interpolatedPrecision = new double[recallLevel.length][numOfLabels];
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            Collections.sort(confact[labelIndex], Collections.reverseOrder());
            double retrievedCounter = 0;
            double relevantCounter = 0;
            double totalRelevantCounter = 0;
            //gia kathe label vriskoume to precision kai vriskoume kai to synoliko arithmo twn sxetikwn
            for (int i = 0; i < confact[labelIndex].size(); i++) {
                retrievedCounter++;
                Boolean actual = confact[labelIndex].get(i).getActual();
                if (actual) {
                    relevantCounter++;

                }
                precision[i][labelIndex] = relevantCounter / retrievedCounter;
                totalRelevantCounter = relevantCounter;
                //System.out.println(labelIndex+"-----"+totalRelevantCounter+"-----"+i+" Precision:"+precision[i][labelIndex]);
            }
            //gia kathe label vriskoume ta recall
            relevantCounter = 0;
            for (int i = 0; i < confact[labelIndex].size(); i++) {
                Boolean actual = confact[labelIndex].get(i).getActual();
                if (actual) {
                    relevantCounter++;
                }
                recall[i][labelIndex] = relevantCounter / totalRelevantCounter;
                //System.out.println(i+" Label index:"+labelIndex+"Precision:"+precision[i][labelIndex]+" recall:"+recall[i][labelIndex]);
            }

        }
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            int pos;
            double sum = 0;
            for (int i = 0; i < recallLevel.length; i++) {
                pos = this.FindRecallPosition(i, labelIndex);
                interpolatedPrecision[i][labelIndex] = this.maxPrecision(pos, labelIndex);
                //System.out.println("recall level"+i+"Label index"+labelIndex+"Interpolated Precision"+interpolatedPrecision[i][labelIndex]);
                sum += interpolatedPrecision[i][labelIndex];
            }
            averageInterpolatedPrecision[labelIndex] = sum / recallLevel.length;
        }        
    }
    
    @Override
    public double getValue() {
        calculateAverageInterpolatedPrecisions();
        return Utils.mean(averageInterpolatedPrecision);
    }

    /**
     * Returns the average interpolated precision for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the average interpolated precision for the given label
     */
    public double getValue(int labelIndex) {
        return averageInterpolatedPrecision[labelIndex];
    }

    @Override
    public double getIdealValue() {
        return 1;
    }

    private double maxPrecision(int pos, int LabelIndex) {
        double maxPrecision = precision[pos][LabelIndex];

        for (int i = pos + 1; i < precision.length; i++) {


            if (precision[i][LabelIndex] > maxPrecision) {
                maxPrecision = precision[i][LabelIndex];
            }
        }
        return maxPrecision;
    }

    private int FindRecallPosition(int Level, int labelIndex) {
        int pos = -1;
        int i = 0;
        boolean done = false;
        while (!done && i < recall.length) {
            if (recall[i][labelIndex] >= recallLevel[Level]) {
                done = true;
                pos = i;
            } else {
                i++;
            }

        }
        return pos;
    }
}
