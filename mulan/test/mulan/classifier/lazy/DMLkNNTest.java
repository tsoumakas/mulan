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
package mulan.classifier.lazy;

import junit.framework.Assert;

import org.junit.Test;

/**
 * Unit test routines for {@link DMLkNN}.
 *
 * @author Grigorios Tsoumakas
 * @version 2012.11.24
 */
public class DMLkNNTest extends MultiLabelKNNTest {

    private static final int DEFAULT_numOfNeighbors = 10;
    private static final double DEFAULT_smooth = 1.0;

    @Override
    public void setUp() {
        learner = new DMLkNN();
    }

    @Test
    public void testTestDefaultParameters() {
        Assert.assertEquals(DEFAULT_numOfNeighbors, learner.numOfNeighbors);
        Assert.assertEquals(DEFAULT_smooth, ((DMLkNN) learner).smooth);

        // common tests
        Assert.assertTrue(learner.isUpdatable());
    }
}
