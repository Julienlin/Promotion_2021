// ant.hpp
#ifndef _ANT_HPP_
# define _ANT_HPP_
# include <utility>
# include "pheromone.hpp"
# include "fractal_land.hpp"
# include "basic_types.hpp"

class ant
{
public:
    /**
     * Une fourmi peut être dans deux états possibles : chargée ( elle porte de la nourriture ) ou non chargée
     */
    enum state { unloaded = 0, loaded = 1 };
    ant(const position_t& pos) : m_state(unloaded), m_position(pos)
    {} 
    ant(const ant& a) = default;
    ant( ant&& a ) = default;
    ~ant() = default;

    void set_loaded() { m_state = loaded; }
    void unset_loaded() { m_state = unloaded; }

    bool is_loaded() const { return m_state == loaded; }
    const position_t& get_position() const { return m_position; }
    static void set_exploration_coef(double eps) { m_eps = eps; }

    void advance( pheromone& phen, const fractal_land& land,
                  const position_t& pos_food, const position_t& pos_nest, std::size_t& cpteur_food );

        ant(state state, position_t pos): m_state(state), m_position(pos){}

        ant& swap(ant &a){
            state buff = m_state;
            m_state = a.m_state;
            a.m_state = buff;
            m_position.swap(a.m_position);
            return *this;
        }
private:
    static double m_eps; // Coefficient d'exploration commun à toutes les fourmis.
    state m_state;
    position_t m_position;
};

#endif
