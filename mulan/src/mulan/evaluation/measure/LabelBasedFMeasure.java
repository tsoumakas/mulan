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
 *    LabelBasedFMeasure.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import mulan.core.MulanRuntimeException;

/**
 * Implementation of the label-based macro precision measure.
 * 
 * @author Grigorios Tso
 * 
 */
public abstract class LabelBasedFMeasure extends BipartitionMeasureBase {

    protected double beta = 1;
    protected int numOfLabels;
    protected double[] falsePositives;
    protected double[] truePositives;
    protected double[] falseNegatives;

    public LabelBasedFMeasure(int numOfLabels) {
        this.numOfLabels = numOfLabels;
        falsePositives = new double[numOfLabels];
        truePositives = new double[numOfLabels];
        falseNegatives = new double[numOfLabels];
    }

    public void reset() {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            falsePositives[labelIndex] = 0;
            truePositives[labelIndex] = 0;
            falseNegatives[labelIndex] = 0;
        }
    }

    public double getIdealValue() {
        return 1;
    }

    protected double calculateFMeasure(double precision, double recall) {
        if ((beta * beta * precision + recall) == 0) {
            reset();
            throw new MulanRuntimeException("F Measure is undefined");
        }
        return ((1 + beta * beta) * precision * recall) / (beta * beta * precision + recall);
    }

    public double updateInternal2(boolean[] bipartition, boolean[] truth) {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            boolean actual = truth[labelIndex];
            boolean predicted = bipartition[labelIndex];

            if (actual && predicted) {
                truePositives[labelIndex]++;
            }
            if (!actual && predicted) {
                falsePositives[labelIndex]++;
            }
            if (actual && !predicted) {
                falseNegatives[labelIndex]++;
            }
        }

        return 0;
    }
}
