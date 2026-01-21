import numpy as np
import scipy as sp
import pygame
from astropy import constants as const

# -------------------------------
# Fonctions utilitaires
# -------------------------------

def barycentre(p1, p2):
    """
    Calcule le barycentre entre deux objets.

    Paramètres :
    - p1, p2 : dictionnaires contenant les clés "mass" et "position" (tuple x, y).

    Retourne :
    - Tuple (x, y) représentant les coordonnées du barycentre.
    """
    m1, m2 = p1["mass"], p2["mass"]
    x1, y1 = p1["position"]
    x2, y2 = p2["position"]
    return (
        (m1 * x1 + m2 * x2) / (m1 + m2),
        (m1 * y1 + m2 * y2) / (m1 + m2)
    )

def distance(p1, p2):
    """
    Calcule la distance entre deux objets.

    Paramètres :
    - p1, p2 : dictionnaires contenant la clé "position" (tuple x, y).

    Retourne :
    - Distance entre les deux objets (float).
    """
    dx = p1["position"][0] - p2["position"][0]
    dy = p1["position"][1] - p2["position"][1]
    return np.hypot(dx, dy)
    
def vitesse_orbitale(planete, etoile):
    """
    Calcule la vitesse orbitale initiale d'une planète autour d'une étoile.

    Paramètres :
    - planete : dictionnaire contenant les clés "position" et "mass".
    - etoile : dictionnaire contenant la clé "mass".

    Retourne :
    - Tuple (vx, vy) représentant les composantes de la vitesse orbitale.
    """
    x0, y0 = planete["position"]
    r0 = np.sqrt(x0**2 + y0**2)
    m = planete["mass"]
    M = etoile["mass"]
    v = np.sqrt(const.G.value * (M + m) / r0)
    vx = -y0 / r0 * v
    vy =  x0 / r0 * v
    return vx, vy

def display_name(screen, obj, position, font=None):
    """
    Affiche le nom d'un objet sur l'écran.

    Paramètres :
    - screen : surface Pygame où afficher le texte.
    - obj : dictionnaire contenant les clés "name" et "color".
    - position : tuple (x, y) pour la position du texte.
    - font : police de caractères (optionnel).
    """
    if font is None:
        font = pygame.font.Font(None, 20)
    text = font.render(obj["name"], True, obj["color"])
    rect = text.get_rect(center=position)
    screen.blit(text, rect)

def equations_mouvement(t, state, planetes, etoile):
    """
    Définit les équations différentielles pour le système N-corps.

    Paramètres :
    - state : tableau numpy contenant les positions et vitesses des planètes.
    - t : temps (non utilisé ici, mais requis pour odeint).
    - planetes : liste de dictionnaires représentant les planètes.
    - etoile : dictionnaire représentant l'étoile.

    Retourne :
    - Tableau numpy contenant les dérivées des positions et vitesses.
    """
    G = const.G.value
    M_star = etoile["mass"]
    n = len(planetes)
    dstate = np.zeros_like(state)
    for i in range(n):
        idx = 4 * i
        xi, yi, vxi, vyi = state[idx:idx+4]
        r = np.hypot(xi, yi)
        axi = -G * M_star * xi / r**3
        ayi = -G * M_star * yi / r**3
        for j in range(n):
            if i == j:
                continue
            idxj = 4 * j
            xj, yj = state[idxj], state[idxj+1]
            dx, dy = xi - xj, yi - yj
            rij = np.hypot(dx, dy)
            if rij == 0: 
                continue
            axi += -G * planetes[j]["mass"] * dx / rij**3
            ayi += -G * planetes[j]["mass"] * dy / rij**3
        dstate[idx:idx+4] = vxi, vyi, axi, ayi
    return dstate

# -------------------------------
# Gestion des boutons
# -------------------------------

def draw_buttons(screen, planetes):
    """
    Dessine les boutons pour sélectionner les planètes.

    Paramètres :
    - screen : surface Pygame où dessiner les boutons.
    - planetes : liste de dictionnaires représentant les planètes.

    Retourne :
    - Liste de tuples (rect, planète) pour chaque bouton.
    """
    buttons = []
    x0, y0 = 50, 120
    w, h = 100, 25
    padding = 5
    font = pygame.font.Font(None, 20)
    for i, p in enumerate(planetes):
        rect = pygame.Rect(x0, y0 + i*(h + padding), w, h)
        pygame.draw.rect(screen, p["color"], rect)
        text_surf = font.render(p["name"], True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)
        buttons.append((rect, p))
    return buttons

def check_button_click(buttons, mouse_pos):
    """
    Vérifie si un bouton a été cliqué.

    Paramètres :
    - buttons : liste de tuples (rect, planète) pour chaque bouton.
    - mouse_pos : position de la souris (tuple x, y).

    Retourne :
    - Planète associée au bouton cliqué, ou None si aucun bouton n'est cliqué.
    """
    for rect, p in buttons:
        if rect.collidepoint(mouse_pos):
            return p
    return None

# -------------------------------
# Simulation et affichage
# -------------------------------

def play_orbit(planetes_dict, etoile_dict, dt=86400, func_zone_habitable=None, func_rayon_influence=None):
    """
    Lance la simulation du système stellaire N-corps.

    Paramètres :
    - planetes_dict : liste de dictionnaires représentant les planètes.
    - etoile_dict : dictionnaire représentant l'étoile.
    - dt : pas de temps en secondes (par défaut : 86400 secondes, soit 1 jour).
    - func_zone_habitable : fonction prenant en entrée le dictionnaire de l'étoile
      et retournant un tuple (r_int, r_ext) des rayons intérieur et extérieur
      de la zone habitable en mètres (optionnel).
    - func_rayon_influence : fonction prenant en entrée le dictionnaire d'une planète
      et retournant le rayon d'influence en mètres (optionnel).

    Cette fonction gère l'affichage interactif, les sliders pour ajuster le zoom
    et le pas de temps, ainsi que le suivi des planètes.
    """

    pygame.init()
    try:
        WIDTH, HEIGHT = 1200, 900
        SCALE = 2e-10
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Système stellaire N-corps")
        clock = pygame.time.Clock()
        planetes = planetes_dict.copy()
        etoile = etoile_dict.copy()
        
        # ---------------------------
        # Slider temps
        # ---------------------------
        slider_dt_x, slider_dt_y = 50, 70
        slider_dt_w, slider_dt_h = 200, 10
        time_slider_color = (180, 180, 180)
        time_handle_color = (100, 255, 100)
        time_handle_radius = 8
        dt_min, dt_max = 3600, 3600*24*7  # 1h à 100 jours par pas
        if dt > dt_max:
            dt = dt_max
        elif dt < dt_min:
            dt = dt_min
        dt_target = np.copy(dt) # valeur actuelle du slider
        dragging_time_slider = False

        # Slider zoom
        slider_zoom_x, slider_zoom_y = 50, 30
        slider_zoom_w, slider_zoom_h = 200, 10
        handle_radius = 8
        slider_color = (200,200,200)
        handle_color = (255,100,100)
        handle_radius = 8
        zoom_min, zoom_max = 0.5, 80.0
        zoom = 1.0
        zoom_target = zoom
        dragging_slider = False

        camera_center = np.array([0.0,0.0])
        prev_camera_center = camera_center.copy()
        camera_target = None  # Planète à suivre, None = Soleil
        targets = planetes + [etoile]

        # Etat initial pour odeint
        state = []
        for p in planetes:
            p["trail"] = []
            p["speed"] = list(vitesse_orbitale(p, etoile))
            state += p["position"]
            state += p["speed"]
        state = np.array(state, dtype=float)


        # Zone habitable
        r_int_pix, r_ext_pix = None, None
        if func_zone_habitable is not None:
            r_int, r_ext = func_zone_habitable(etoile)
            # Position de l'étoile
            xs, ys = etoile["position"]
            # Conversion en coordonnées écran
            sx_star = int((xs - camera_center[0]) * SCALE * zoom + WIDTH/2)
            sy_star = int((ys - camera_center[1]) * SCALE * zoom + HEIGHT/2)

            # Rayons en pixels
            r_int_pix = int(r_int * SCALE * zoom)
            r_ext_pix = int(r_ext * SCALE * zoom)

        # Rayon influence planétaires
        if func_rayon_influence is not None:
            for p in planetes:
                r_inf = func_rayon_influence(p, etoile)
                p["rayon_influence"] = r_inf

        running = True
        while running:

            # ---------------------------
            # Événements
            # ---------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Slider zoom
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    handle_x = slider_zoom_x + (zoom_target - zoom_min)/(zoom_max-zoom_min)*slider_zoom_w
                    if np.hypot(mx-handle_x, my-(slider_zoom_y+slider_zoom_h//2)) < handle_radius:
                        dragging_slider = True
                    # Boutons
                    clicked_target = check_button_click(draw_buttons(screen, targets), (mx,my))
                    if clicked_target:
                        camera_target = clicked_target

                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_slider = False
                if event.type == pygame.MOUSEMOTION and dragging_slider:
                    mx, my = pygame.mouse.get_pos()
                    relative_x = np.clip(mx - slider_zoom_x, 0, slider_zoom_w)
                    zoom_target = zoom_min + (relative_x / slider_zoom_w) * (zoom_max - zoom_min)

                # --- slider temps ---
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    handle_x = slider_dt_x + (dt_target - dt_min)/(dt_max-dt_min)*slider_dt_w
                    if np.hypot(mx-handle_x, my-(slider_dt_y+slider_dt_h//2)) < time_handle_radius:
                        dragging_time_slider = True

                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_time_slider = False

                if event.type == pygame.MOUSEMOTION and dragging_time_slider:
                    mx, my = pygame.mouse.get_pos()
                    relative_x = np.clip(mx - slider_dt_x, 0, slider_dt_w)
                    dt_target = dt_min + (relative_x / slider_dt_w) * (dt_max - dt_min)


            zoom += 0.15 * (zoom_target - zoom)
            dt += 0.2 * (dt_target - dt)

            # ---------------------------
            # Calcul physique
            # ---------------------------
            tspan = [1e-10, dt]
            # sol = sp.integrate.odeint(equations_mouvement, state, tspan, args=(planetes, etoile))
            # state = sol[-1]
            sol = sp.integrate.solve_ivp(equations_mouvement, tspan, state, args=(planetes, etoile), method='RK45', rtol=1e-6, atol=1e-8)
            state = sol.y[:, -1]
            #
            for i, p in enumerate(planetes):
                idx = 4*i
                p["position"][0] = state[idx]
                p["position"][1] = state[idx+1]
                p["speed"][0] = state[idx+2]
                p["speed"][1] = state[idx+3]

            # ---------------------------
            # Caméra
            # ---------------------------
            if camera_target is None:
                camera_center[:] = (0.0,0.0)  # système global
            else:
                camera_center[:] = camera_target["position"]

            screen.fill((0,0,0))

            # Slider zoom
            pygame.draw.rect(screen, slider_color, (slider_zoom_x, slider_zoom_y, slider_zoom_w, slider_zoom_h))
            handle_x = slider_zoom_x + (zoom_target-zoom_min)/(zoom_max-zoom_min)*slider_zoom_w
            handle_y = slider_zoom_y + slider_zoom_h//2
            pygame.draw.circle(screen, handle_color, (int(handle_x), int(handle_y)), handle_radius)

            # valeur zoom
            font = pygame.font.Font(None, 20)
            text_zoom = font.render(f"Zoom: {zoom:.2f}x", True, (255,255,255))
            screen.blit(text_zoom, (slider_zoom_x+slider_zoom_w+10, slider_zoom_y-5))

            # Zone habitable
            if func_zone_habitable is not None:
                try: 
                    r_int, r_ext = func_zone_habitable(etoile)
                    # Position de l'étoile en coordonnées écran
                    sx_star = int((etoile["position"][0] - camera_center[0]) * SCALE * zoom + WIDTH/2)
                    sy_star = int((etoile["position"][1] - camera_center[1]) * SCALE * zoom + HEIGHT/2)
                    # Rayons en pixels
                    r_int_pix = int(r_int * SCALE * zoom)
                    r_ext_pix = int(r_ext * SCALE * zoom)
                except Exception as e:
                    r_int_pix, r_ext_pix = None, None   
                    raise ArithmeticError("Erreur dans la fonction de la zone habitable : ", e)

            # Slider temps
            pygame.draw.rect(screen, time_slider_color, (slider_dt_x, slider_dt_y, slider_dt_w, slider_dt_h))
            time_handle_x = slider_dt_x + (dt_target - dt_min)/(dt_max-dt_min)*slider_dt_w
            time_handle_y = slider_dt_y + slider_dt_h//2
            pygame.draw.circle(screen, time_handle_color, (int(time_handle_x), int(time_handle_y)), time_handle_radius)

            # valeur dt
            text_dt = font.render(f"Pas de temps: {dt/86400:.2f} jours", True, (255,255,255))
            screen.blit(text_dt, (slider_dt_x+slider_dt_w+10, slider_dt_y-5))

            # Boutons
            draw_buttons(screen, targets)

            # Planètes et trails
            for p in planetes:
                x, y = p["position"]
                sx = int((x - camera_center[0]) * SCALE * zoom + WIDTH/2)
                sy = int((y - camera_center[1]) * SCALE * zoom + HEIGHT/2)

                p["trail"].append((x,y))
                if len(p["trail"])>400:
                    p["trail"].pop(0)
                trail_screen = [(int((tx-camera_center[0])*SCALE*zoom+WIDTH/2),
                                int((ty-camera_center[1])*SCALE*zoom+HEIGHT/2))
                                for tx, ty in p["trail"]]
                if len(trail_screen)>2:
                    pygame.draw.lines(screen, p["color"], False, trail_screen, 1)

                pygame.draw.circle(screen, p["color"], (sx,sy), p["apparent_size"])
                display_name(screen, p, (sx,sy - p["apparent_size"]-12))

            # Étoile
            xs, ys = etoile["position"]
            sx_star = int((xs - camera_center[0]) * SCALE * zoom + WIDTH/2)
            sy_star = int((ys - camera_center[1]) * SCALE * zoom + HEIGHT/2)
            pygame.draw.circle(screen, etoile["color"], (sx_star, sy_star), etoile["apparent_size"])
            display_name(screen, etoile, (sx_star, sy_star - etoile["apparent_size"] - 15))

            if (r_int_pix is not None) and (r_ext_pix is not None):
                # Zone habitable
                # Dessiner l'anneau de la zone habitable (en vert translucide)
                color_zh = (0, 255, 150, 50)
                pygame.draw.circle(screen, color_zh, (sx_star, sy_star), r_ext_pix, 2)  # 2 is the width of the outline
                pygame.draw.circle(screen, color_zh, (sx_star, sy_star), r_int_pix, 2) #, 10)            
                zone_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                zone_surface.set_alpha(128)
                pygame.draw.circle(zone_surface, color_zh, (sx_star, sy_star), r_ext_pix)
                pygame.draw.circle(zone_surface, (0, 0, 0, 0), (sx_star, sy_star), r_int_pix)
                screen.blit(zone_surface, (0, 0))

                # Affichage du texte "Zone Habitable"
                zh_text = font.render("Zone Habitable", True, color_zh)
                zh_rect = zh_text.get_rect(center=(sx_star, sy_star - r_ext_pix - 10))
                screen.blit(zh_text, zh_rect)

            if func_rayon_influence is not None:
                # Rayons d'influence
                for p in planetes:
                    r_inf = p.get("rayon_influence", None)
                    if r_inf is not None:
                        r_inf_pix = int(r_inf * SCALE * zoom)
                        x, y = p["position"]
                        sx = int((x - camera_center[0]) * SCALE * zoom + WIDTH/2)
                        sy = int((y - camera_center[1]) * SCALE * zoom + HEIGHT/2)
                        # Dessiner le cercle de la zone d'influence (en bleu translucide)
                        color_inf = (100, 100, 255, 50)
                        pygame.draw.circle(screen, color_inf, (sx, sy), r_inf_pix, 2)  # 2 is the width of the outline
                        inf_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                        inf_surface.set_alpha(128)
                        pygame.draw.circle(inf_surface, color_inf, (sx, sy), r_inf_pix)
                        screen.blit(inf_surface, (0, 0))

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.display.quit()
