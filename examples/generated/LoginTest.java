package org.eclipse.swtbot.test;

import static org.junit.Assert.*;
import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.eclipse.finder.SWTWorkbenchBot;
import org.eclipse.swtbot.swt.finder.junit.SWTBotJunit4ClassRunner;
import org.eclipse.swtbot.swt.finder.utils.SWTBotPreferences;
import org.eclipse.swtbot.swt.finder.widgets.*;
import org.eclipse.swtbot.swt.finder.exceptions.WidgetNotFoundException;
import org.eclipse.swtbot.swt.finder.waits.Conditions;
import org.eclipse.swtbot.swt.finder.waits.DefaultCondition;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Auto-generated test from AI Test Script Generator
 * Test the login functionality
 */
@RunWith(SWTBotJunit4ClassRunner.class)
public class LoginTest {
    
    private SWTWorkbenchBot bot;
    
    @Before
    public void setUp() throws Exception {
        // Configure SWTBot preferences
        SWTBotPreferences.PLAYBACK_DELAY = 100;
        SWTBotPreferences.TIMEOUT = 10000;
        
        // Initialize bot
        bot = new SWTWorkbenchBot();
        
        // Close welcome view if it exists
        try {
            bot.viewByTitle("Welcome").close();
        } catch (WidgetNotFoundException e) {
            // Welcome view not found, ignore
        }
    }
    
    @Test
    public void login() throws Exception {
        // Click button 'Login'
        bot.button("Login").click();
        
        // Enter 'admin' in the username field
        bot.textWithLabel("Username:").setText("admin");
        
        // Enter 'password123' in the password field
        bot.textWithLabel("Password:").setText("password123");
        
        // Click button 'Submit'
        bot.button("Submit").click();
        
        // Verify welcome message
        assertTrue("Welcome message should be displayed", 
            bot.label().getText().contains("Welcome"));
    }
    
    @After
    public void tearDown() throws Exception {
        // Clean up any open dialogs
        while (tryCloseShell());
    }
    
    /**
     * Helper method to close any open dialogs
     */
    private boolean tryCloseShell() {
        try {
            SWTBotShell shell = bot.activeShell();
            if (shell != null && shell.isOpen() && !shell.isActive()) {
                shell.close();
                return true;
            }
        } catch (Exception e) {
            // Ignore exceptions
        }
        return false;
    }
}
