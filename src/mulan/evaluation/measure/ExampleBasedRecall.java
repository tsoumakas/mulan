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
 *    ExampleBasedRecall.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

/**
 * Implementation of the example-based recall measure.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class ExampleBasedRecall extends ExampleBasedBipartitionMeasureBase {

    private final boolean strict;

    public String getName() {
        return "Example-Based Recall";
    }

    public double getIdealValue() {
        return 1;
    }

    /**
     * Creates a new object
     *
     * @param strict when false, divisions-by-zero are ignored
     */
    public ExampleBasedRecall(boolean strict) {
        this.strict = strict;
    }

    @Override
    protected void updateBipartition(boolean[] bipartition, boolean[] truth) {
        double intersection = 0;
        double actual = 0;
        for (int i = 0; i < truth.length; i++) {
            if (truth[i]) {
                actual++;
                if (bipartition[i]) {
                    intersection++;
                }
            }
        }

        if (actual == 0 && strict == false)
            return;

        sum += intersection / actual;
        count++;
    }
}
