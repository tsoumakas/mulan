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
package mulan.core;

/**
 * The convenience exception, which can be used to wrap up checked general {@link Exception}
 * commonly thrown by underlying Weka library into anonymous runtime exception.
 * <br><br> 
 * Note: The preferred way of handling Weka exceptional states is to define custom typed 
 * exception thrown by Mulan, which specifies a context about failure reason. 
 * 
 * @author Jozef Vilcek
 */
public class WekaException extends MulanRuntimeException {

    private static final long serialVersionUID = -8041689691825060987L;

    /**
     * Creates a new instance of {@link WekaException} with detail mesage.
     * @param message the detail message
     */
    public WekaException(String message) {
        super(message);
    }

    /**
     * Creates a new instance of {@link WekaException} with detail message
     * and nested exception.
     *
     * @param message the detail message
     * @param cause the nested exception
     */
    public WekaException(String message, Throwable cause) {
        super(message, cause);
    }
}