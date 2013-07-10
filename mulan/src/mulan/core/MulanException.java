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
 * Represents a root base class for checked exceptions thrown within Mulan library.
 * 
 * @author Jozef Vilcek
 */
public class MulanException extends Exception {

    private static final long serialVersionUID = 2271544759439172440L;

    /**
     * Creates a new instance of {@link MulanException} with the specified
     * detail message.
     *
     * @param message the detail message
     */
    public MulanException(String message) {
        super(message);
    }

    /**
     * Creates a new instance of {@link MulanException} with the specified
     * detail message and nested exception.
     *
     * @param message the detail message
     * @param cause the nested exception
     */
    public MulanException(String message, Throwable cause) {
        super(message, cause);
    }
}