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
 *    ExampleBasedFMeasure.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

/**
 * Implementation of the example-based F measure.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class ExampleBasedFMeasure extends ExampleBasedBipartitionMeasureBase {

    private final double beta;
    private final boolean strict;

    /**
     * Creates a new object
     *
     * @param strict when false, divisions-by-zero are ignored
     */
    public ExampleBasedFMeasure(boolean strict) {
        this(strict, 1.0);
    }

    /**
     * Creates a new object
     *
     * @param strict when false, divisions-by-zero are ignored
     * @param beta the beta parameter for precision and recall combination
     */
    public ExampleBasedFMeasure(boolean strict, double beta) {
        this.strict = strict;
        this.beta = beta;
    }

    public String getName() {
        return "Example-Based F Measure";
    }

    public double getIdealValue() {
        return 1;
    }

    @Override
    protected void updateBipartition(boolean[] bipartition, boolean[] truth) {
        double intersection = 0;
        double predicted = 0;
        double actual = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i]) {
                predicted++;
            }
            if (truth[i]) {
                actual++;
            }
            if (bipartition[i] && truth[i]) {
                intersection++;
            }
        }
        if ((predicted == 0 || actual == 0) && (strict == false))
            return;

        double precision = intersection / predicted;
        double recall = intersection / actual;

        if (((beta * beta * precision + recall) == 0) && (strict == false))
            return;

        sum += ((1 + beta * beta) * precision * recall) / (beta * beta * precision + recall);
        count++;
    }
}
