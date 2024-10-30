
// TOP-DOWN FPS ENVIRONMENT
// PROOF OF CONCEPT


// Include libraries
#include <SFML/Graphics.hpp>
#include <cmath>
#include <vector>
#include <random>


// Window dimensions
const int WIDTH = 800;
const int HEIGHT = 600;

// Agent settings
const float AGENT_SIZE = 40.0f;
const float AGENT_SPEED = 2.0f;

// Bullet settings
const float BULLET_RADIUS = 5.0f;
const float BULLET_SPEED = 10.0f;


// Bullet behaviour
struct Bullet {
    sf::CircleShape shape;
    sf::Vector2f velocity;

    Bullet(float x, float y, float angle) {
        shape.setRadius(BULLET_RADIUS);
        shape.setFillColor(sf::Color::Red);
        shape.setPosition(x, y);
        velocity.x = BULLET_SPEED*std::cos(angle);
        velocity.y = BULLET_SPEED*std::sin(angle);
    }

    void update() {
        shape.move(velocity);
    }
};


// Main
int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Basic Shooter");

    // Agent setup
    sf::RectangleShape agent(sf::Vector2f(AGENT_SIZE, AGENT_SIZE));
    agent.setFillColor(sf::Color::Black);
    agent.setPosition(WIDTH/2, HEIGHT/2);

    // Bullets
    std::vector<Bullet> bullets;

    // Game loop
    while (window.isOpen()) {

        // Check if window is closed
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Agent movement
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W) && agent.getPosition().y > 0)
            agent.move(0, -AGENT_SPEED);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S) && agent.getPosition().y < HEIGHT - AGENT_SIZE)
            agent.move(0, AGENT_SPEED);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A) && agent.getPosition().x > 0)
            agent.move(-AGENT_SPEED, 0);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D) && agent.getPosition().x < WIDTH - AGENT_SIZE)
            agent.move(AGENT_SPEED, 0);

        // Shooting bullets
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2f agentCenter = agent.getPosition() + sf::Vector2f(AGENT_SIZE/2, AGENT_SIZE/2);
            sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
            float angle = std::atan2(mousePos.y - agentCenter.y, mousePos.x - agentCenter.x);
            bullets.emplace_back(agentCenter.x, agentCenter.y, angle);
        }
    
        // Update bullets
        for (auto& bullet : bullets)
            bullet.update();

        // Render everything
        window.clear(sf::Color::White);
        window.draw(agent);
        for (const auto& bullet : bullets)
            window.draw(bullet.shape);
        window.display();
    }

    return 0;
}
