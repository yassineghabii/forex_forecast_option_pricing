package com.finlogik.back.forum.entities;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.*;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.Date;
import java.util.Set;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class Topic implements Serializable{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idTopic;

    private String title;
    private String question;
    private Integer likes;
    private Integer dislikes;

    @Temporal(TemporalType.TIMESTAMP)
    private Date creationDate;
    private String userId;
    @OneToMany(mappedBy = "topic", cascade = CascadeType.ALL)
    @JsonIgnore
    private Set<Post> posts;
}
