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
package mulan.evaluation.measure;

/**
 * Base class for example-based bipartition measures
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.03
 */
public abstract class ExampleBasedBipartitionMeasureBase extends BipartitionMeasureBase {

    /**
     * The current sum of the measure
     */
    protected double sum;
    /**
     * The number of validation examples processed
     */
    protected int count;

    @Override
    public void reset() {
        sum = 0;
        count = 0;
    }

    @Override
    public double getValue() {
        return sum / count;
    }

}